# tests/test_env.py
import pytest
from sere.core.pddl_env.env import PDDLEnv
from sere.core.world_state import WorldState
from sere.pddl.domain_spec import (
    DomainSpec,
    ActionSpec,
    PredicateSpec,
    FluentSpec,
    OutcomeSpec,
    ConditionalBlock,
)


@pytest.fixture
def make_env():
    def _make_env(actions=None, predicates=None, fluents=None, static_facts=None, **kwargs):
        domain = DomainSpec(
            name="dummy",
            types={},
            predicates=predicates or {},
            actions=actions or {},
            fluents=fluents or {},
        )
        world = WorldState(domain=domain, objects={}, facts=set(), fluents={})
        return PDDLEnv(
            domain=domain,
            world=world,
            static_facts=static_facts or set(),
            max_steps=5,
            step_penalty=-0.01,
            invalid_penalty=-1.0,
            **kwargs,
        )
    return _make_env


def test_reset_clears_and_initializes(make_env):
    env = make_env()
    obs, info = env.reset(seed=123)
    assert isinstance(obs, str) and obs
    assert "system_prompt" in info
    assert env.steps == 0
    assert env.time == 0.0
    assert not env.done
    assert env.messages == []
    assert env._step_messages == []


def test_retry_accumulation_and_penalties(make_env):
    env = make_env()
    env.illegal_move_retries = 2
    env.invalid_retry_penalty = -0.05
    env.reset()

    obs, r, done, info = env.step("<move>(foo a b)</move>")
    assert not done and r == -0.05 and info["outcome"] == "invalid_move"

    obs, r, done, info = env.step("<move>(foo a b)</move>")
    assert not done and r == -0.05 and info["outcome"] == "invalid_move"

    obs, r, done, info = env.step("<move>(foo a b)</move>")
    assert done and r == -1.0 and info["outcome"] == "invalid_move"


def test_step_penalty_and_termination_rule(make_env):
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    # Use a static fact in the termination rule; evaluator treats plain literals as true if present.
    env = make_env(
        actions={"noop": a},
        termination_rules=[{"when": "(always)", "reward": 5, "outcome": "success"}],
        static_facts={("always", ())},
    )
    env.reset()

    obs, r, done, info = env.step("<move>(noop)</move>")
    assert done
    assert info["outcome"] == "success"
    assert r == pytest.approx(-0.01 + 5.0)


def test_numeric_clamping(make_env):
    fl = FluentSpec(name="energy", args=[("r", "robot")], nl=["energy"])
    cap = FluentSpec(name="battery-cap", args=[("r", "robot")], nl=["battery cap"])
    env = make_env(fluents={"energy": fl, "battery-cap": cap}, enable_numeric=True)
    env.reset()
    r = "r1"
    env.world.objects[r] = {"robot"}

    env.world.set_fluent("energy", (r,), -5.0)
    env._enforce_energy_bounds()
    assert env.world.get_fluent("energy", (r,)) == 0.0

    env.world.set_fluent("battery-cap", (r,), 10.0)
    env.world.set_fluent("energy", (r,), 50.0)
    env._enforce_energy_bounds()
    assert env.world.get_fluent("energy", (r,)) == 10.0


def test_stochastic_outcomes_respect_rng(make_env):
    oc1 = OutcomeSpec(name="success", p=1.0)
    oc2 = OutcomeSpec(name="fail", p=1.0)
    a = ActionSpec(name="act", params=[], pre=[], add=[], delete=[], nl=["act"], outcomes=[oc1, oc2])
    env = make_env(actions={"act": a}, enable_stochastic=True, seed=42)
    env.reset()

    # Same seed → same outcome branch across resets
    branches = []
    for _ in range(3):
        obs, r, done, info = env.step("<move>(act)</move>")
        branches.append(info.get("outcome_branch"))
        env.reset(seed=42)
    assert len(set(branches)) == 1
    assert branches[0] in {"success", "fail"}


def test_conditional_effects(make_env):
    a = ActionSpec(name="move", params=[("r", "robot"), ("l", "loc")], pre=[], add=[], delete=[], nl=["move"], cond=[])
    a.cond = [ConditionalBlock(when=["(at r1 kitchen)"], add=["(happy)"], delete=[], num_eff=[], messages=[])]
    env = make_env(actions={"move": a}, enable_conditional=True)
    env.world.facts.add(("at", ("r1", "kitchen")))
    env.reset()

    obs, r, done, info = env.step("<move>(move r1 kitchen)</move>")
    assert ("happy", ()) in env.world.facts


def test_renderer_messages_ephemeral(make_env):
    a = ActionSpec(name="say", params=[], pre=[], add=[], delete=[], nl=["say"], messages=["hello"])
    env = make_env(actions={"say": a})
    env.reset()

    obs1, r, done, info1 = env.step("<move>(say)</move>")
    assert "hello" in info1["messages"]

    obs2, r, done, info2 = env.step("<move>(say)</move>")
    assert "hello" in info2["messages"]
    # Each step exposes only last-turn messages
    assert info1["messages"].count("hello") == 1
    assert info2["messages"].count("hello") == 1


def test_invalid_report_contains_human_text(make_env):
    a = ActionSpec(name="needs-fact", params=[], pre=["(missing-fact)"], add=[], delete=[], nl=["needs fact"])
    env = make_env(actions={"needs-fact": a})
    env.reset()

    obs, r, done, info = env.step("<move>(needs-fact)</move>")
    assert "Preconditions were not satisfied" in obs


def test_open_loop_plan_executes_in_sequence(make_env):
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(actions={"noop": a})
    env.reset()

    obs, r, done, info = env.step("<move>(noop)(noop)(noop)</move>")
    trace = info["plan_trace"]
    assert len(trace) == 3
    assert all(x["action"] == "noop" for x in trace)


def test_open_loop_atomic_rolls_back(make_env):
    a = ActionSpec(name="bad", params=[], pre=["(missing)"], add=[], delete=[], nl=["bad"])
    env = make_env(actions={"bad": a})
    env.reset()

    from sere.core.pddl_env.planning import execute_plan, parse_move_block
    plan = parse_move_block("<move>(bad)</move>")
    obs, r, done, info = execute_plan(env, plan, atomic=True)

    assert not env.world.facts
    assert done
    assert info["outcome"] == "invalid_move"