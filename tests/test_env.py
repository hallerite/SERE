# tests/test_env.py
import pytest
from sere.core.pddl_env.env import PDDLEnv
from sere.core.world_state import WorldState
from sere.core.pddl_env.run_mode import RunMode
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
    def _make_env(
        actions=None,
        predicates=None,
        fluents=None,
        static_facts=None,
        types=None,
        *,
        formatter_config=None,
        **kwargs
    ):
        domain = DomainSpec(
            name="dummy",
            types=types or {},
            predicates=predicates or {},
            actions=actions or {},
            fluents=fluents or {},
            derived={},
        )
        world = WorldState(domain=domain, objects={}, facts=set(), fluents={})

        # Ensure footer is rendered in observations for tests that assert on it
        fc = dict(formatter_config or {})
        fc.setdefault("show_footer", True)

        return PDDLEnv(
            domain=domain,
            world=world,
            static_facts=static_facts or set(),
            max_steps=5,
            step_penalty=-0.01,
            invalid_penalty=-1.0,
            formatter_config=fc,
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

    obs, r, done, info = env.step("(foo a b)")
    assert not done and r == -0.05 and info["outcome"] == "invalid_move"

    obs, r, done, info = env.step("(foo a b)")
    assert not done and r == -0.05 and info["outcome"] == "invalid_move"

    obs, r, done, info = env.step("(foo a b)")
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

    obs, r, done, info = env.step("(noop)")
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
        obs, r, done, info = env.step("(act)")
        branches.append(info.get("outcome_branch"))
        env.reset(seed=42)
    assert len(set(branches)) == 1
    assert branches[0] in {"success", "fail"}


def test_conditional_effects(make_env):
    a = ActionSpec(name="move", params=[("r", "robot"), ("l", "loc")], pre=[], add=[], delete=[], nl=["move"], cond=[])
    a.cond = [ConditionalBlock(when=["(at r1 kitchen)"], add=["(happy)"], delete=[], num_eff=[], messages=[])]
    env = make_env(
        actions={"move": a},
        types={"robot": "", "loc": ""},
        enable_conditional=True,
    )
    env.world.objects["r1"] = {"robot"}
    env.world.objects["kitchen"] = {"loc"}
    env.world.facts.add(("at", ("r1", "kitchen")))
    env.reset()

    obs, r, done, info = env.step("(move r1 kitchen)")
    assert ("happy", ()) in env.world.facts


def test_static_predicates_cannot_change(make_env):
    adj = PredicateSpec(name="adjacent", args=[], nl=["adjacent"], static=True)
    a = ActionSpec(name="bad", params=[], pre=[], add=["(adjacent)"], delete=[], nl=["bad"])
    env = make_env(actions={"bad": a}, predicates={"adjacent": adj})
    env.reset()

    obs, r, done, info = env.step("(bad)")
    assert info.get("outcome") == "invalid_move"
    assert "static" in info.get("error", "").lower()


def test_joint_actions_simultaneous_swap(make_env):
    at = PredicateSpec(name="at", args=[("r", "robot"), ("l", "loc")], nl=["{r} at {l}"])
    move = ActionSpec(
        name="move",
        params=[("r", "robot"), ("from", "loc"), ("to", "loc")],
        pre=["(at ?r ?from)"],
        add=["(at ?r ?to)"],
        delete=["(at ?r ?from)"],
        nl=["move"],
    )
    env = make_env(
        actions={"move": move},
        predicates={"at": at},
        types={"robot": "", "loc": ""},
        run_mode=RunMode.INTERACTIVE,
        multi_agent=True,
    )
    env.world.objects["r1"] = {"robot"}
    env.world.objects["r2"] = {"robot"}
    env.world.objects["A"] = {"loc"}
    env.world.objects["B"] = {"loc"}
    env.world.facts |= {("at", ("r1", "A")), ("at", ("r2", "B"))}
    env.reset()

    obs, r, done, info = env.step("(move r1 A B)(move r2 B A)")
    assert info.get("outcome") != "invalid_move"
    assert ("at", ("r1", "B")) in env.world.facts
    assert ("at", ("r2", "A")) in env.world.facts


def test_joint_actions_require_each_robot(make_env):
    move = ActionSpec(
        name="move",
        params=[("r", "robot"), ("to", "loc")],
        pre=[],
        add=[],
        delete=[],
        nl=["move"],
    )
    env = make_env(
        actions={"move": move},
        types={"robot": "", "loc": ""},
        run_mode=RunMode.INTERACTIVE,
        multi_agent=True,
    )
    env.world.objects["r1"] = {"robot"}
    env.world.objects["r2"] = {"robot"}
    env.world.objects["A"] = {"loc"}
    env.reset()

    obs, r, done, info = env.step("(move r1 A)")
    assert info.get("outcome") == "invalid_move"
    assert "expects 2 actions" in info.get("error", "").lower()


def test_renderer_messages_ephemeral(make_env):
    a = ActionSpec(name="say", params=[], pre=[], add=[], delete=[], nl=["say"], messages=["hello"])
    env = make_env(actions={"say": a})
    env.reset()

    obs1, r, done, info1 = env.step("(say)")
    assert "hello" in info1["messages"]

    obs2, r, done, info2 = env.step("(say)")
    assert "hello" in info2["messages"]
    # Each step exposes only last-turn messages
    assert info1["messages"].count("hello") == 1
    assert info2["messages"].count("hello") == 1


def test_invalid_report_contains_human_text(make_env):
    a = ActionSpec(name="needs-fact", params=[], pre=["(missing-fact)"], add=[], delete=[], nl=["needs fact"])
    env = make_env(actions={"needs-fact": a})
    env.reset()

    obs, r, done, info = env.step("(needs-fact)")
    assert "Preconditions were not satisfied" in obs


def test_open_loop_atomic_rolls_back(make_env):
    a = ActionSpec(name="bad", params=[], pre=["(missing)"], add=[], delete=[], nl=["bad"])
    env = make_env(actions={"bad": a})
    env.reset()

    from sere.core.pddl_env.planning import execute_plan, parse_actions
    plan = parse_actions("(bad)")
    obs, r, done, info = execute_plan(env, plan, atomic=True)

    assert not env.world.facts
    assert done
    assert info["outcome"] == "invalid_move"

def test_sysprompt_and_footer_interactive(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(actions={"noop": a}, run_mode=RunMode.INTERACTIVE)
    obs, info = env.reset()
    # System prompt should mention INTERACTIVE
    assert "You are in INTERACTIVE mode." in info["system_prompt"]
    # Observation footer should be single-action hint
    assert "Reply with (action args)." in obs


def test_sysprompt_and_footer_batch(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(actions={"noop": a}, run_mode=RunMode.BATCH)
    obs, info = env.reset()
    assert "You are in BATCH mode." in info["system_prompt"]
    # Observation footer should allow multiple actions
    assert "You may submit multiple actions: (a1 ...)(a2 ...)...." in obs


def test_sysprompt_and_footer_open_loop(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(actions={"noop": a}, run_mode=RunMode.OPEN_LOOP)
    obs, info = env.reset()
    assert "You are in OPEN-LOOP mode." in info["system_prompt"]
    # Observation footer should state episode ends after execution
    assert "Episode will end after execution." in obs


def test_interactive_rejects_multi_action(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(actions={"noop": a}, run_mode=RunMode.INTERACTIVE, illegal_move_retries=0)
    env.reset()

    obs, r, done, info = env.step("(noop)(noop)")
    assert info["outcome"] == "invalid_move"
    assert "expects exactly one action" in info["error"]
    assert done  # retries=0 -> terminal immediately
    assert r == -1.0  # invalid_penalty


def test_batch_all_valid_keeps_episode_open(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(actions={"noop": a}, run_mode=RunMode.BATCH)
    env.reset()

    obs, r, done, info = env.step("(noop)(noop)(noop)")
    assert info.get("plan_mode") == "batch"
    assert info["steps_executed"] == 3
    assert len(info["plan_trace"]) == 3
    assert not done  # batch should not force terminal by itself


def test_batch_partial_abort_reports_reason_and_progress(make_env):
    from sere.core.pddl_env.env import RunMode
    # a1 is valid and adds a fact; a2 requires a missing fact -> invalid
    a1 = ActionSpec(name="addx", params=[], pre=[], add=["(x)"], delete=[], nl=["addx"])
    a2 = ActionSpec(name="needs-missing", params=[], pre=["(missing)"], add=[], delete=[], nl=["needs missing"])
    env = make_env(actions={"addx": a1, "needs-missing": a2}, run_mode=RunMode.BATCH, illegal_move_retries=0)
    env.reset()

    obs, r, done, info = env.step("(addx)(needs-missing)(addx)")
    # First action executed, then aborted on second
    assert ("x", ()) in env.world.facts
    assert info.get("plan_aborted") is True
    assert info.get("aborted_at") == 2
    assert "Preconditions were not satisfied" in (info.get("abort_error") or "")
    assert info["outcome"] == "invalid_move"
    assert done  # retries=0 -> terminal immediately on invalid


def test_open_loop_forces_terminal_even_without_rules(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(actions={"noop": a}, run_mode=RunMode.OPEN_LOOP)
    env.reset()

    obs, r, done, info = env.step("(noop)(noop)")
    assert info.get("plan_mode") == "open_loop"
    assert info["steps_executed"] == 2
    # Should force terminal because open-loop ends after one call
    assert done
    # If nothing else set outcome, reason should be open_loop_end
    assert info.get("reason") in {"open_loop_end", "implicit_fail", "terminal"}  # allow domain to set one


def test_open_loop_invalid_move_is_terminal_and_reports_abort(make_env):
    from sere.core.pddl_env.env import RunMode
    bad = ActionSpec(name="bad", params=[], pre=["(missing)"], add=[], delete=[], nl=["bad"])
    env = make_env(actions={"bad": bad}, run_mode=RunMode.OPEN_LOOP, illegal_move_retries=0)
    env.reset()

    obs, r, done, info = env.step("(bad)(bad)")
    assert done
    assert info["outcome"] == "invalid_move"
    assert info.get("plan_aborted") is True
    assert info.get("aborted_at") == 1  # fails on first action


def test_mode_specific_affordance_footer_text(make_env):
    # This checks the observation footer message varies by mode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])

    # INTERACTIVE
    env_i = make_env(actions={"noop": a})
    obs_i, _ = env_i.reset()
    assert obs_i.strip().endswith("Reply with (action args).")

    # BATCH
    from sere.core.pddl_env.env import RunMode
    env_b = make_env(actions={"noop": a}, run_mode=RunMode.BATCH)
    obs_b, _ = env_b.reset()
    assert "You may submit multiple actions:" in obs_b

    # OPEN_LOOP
    env_o = make_env(actions={"noop": a}, run_mode=RunMode.OPEN_LOOP)
    obs_o, _ = env_o.reset()
    assert "Episode will end after execution." in obs_o


def test_info_fields_present_in_batch(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(actions={"noop": a}, run_mode=RunMode.BATCH)
    env.reset()
    obs, r, done, info = env.step("(noop)(noop)")
    # Plan tracing and shaping totals should be present
    assert isinstance(info.get("plan_trace"), list)
    assert isinstance(info.get("steps_executed"), int)
    assert "shaping_bonus_total" in info
    assert info.get("plan_mode") == "batch"


def test_interactive_single_action_executes_and_continues(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(actions={"noop": a}, run_mode=RunMode.INTERACTIVE)
    env.reset()
    obs, r, done, info = env.step("(noop)")
    assert info["outcome"] in {"ongoing", "success", "timeout", "failed"}  # normalized by env
    assert not done or info["outcome"] in {"success", "timeout"}  # typically not done after one noop

def test_interactive_multi_action_does_not_mutate_state(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="addx", params=[], pre=[], add=[("x", ())], delete=[], nl=["addx"])
    env = make_env(actions={"addx": a}, run_mode=RunMode.INTERACTIVE, illegal_move_retries=0)
    env.reset()
    obs, r, done, info = env.step("(addx)(addx)")
    assert ("x", ()) not in env.world.facts
    assert info["outcome"] == "invalid_move"
    assert done

def test_batch_partial_abort_does_not_apply_later_actions(make_env):
    from sere.core.pddl_env.env import RunMode
    ok  = ActionSpec(name="ok",  params=[], pre=[], add=["(a)"], delete=[], nl=["ok"])
    bad = ActionSpec(name="bad", params=[], pre=["(missing)"], add=["(b)"], delete=[], nl=["bad"])
    ok2 = ActionSpec(name="ok2", params=[], pre=[], add=["(c)"], delete=[], nl=["ok2"])
    env = make_env(actions={"ok": ok, "bad": bad, "ok2": ok2}, run_mode=RunMode.BATCH, illegal_move_retries=0)
    env.reset()
    obs, r, done, info = env.step("(ok)(bad)(ok2)")
    assert ("a", ()) in env.world.facts
    assert ("b", ()) not in env.world.facts
    assert ("c", ()) not in env.world.facts
    assert info.get("plan_aborted") is True and info.get("aborted_at") == 2

def test_open_loop_respects_success_rule_without_extra_reason(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])
    env = make_env(
        actions={"noop": a},
        run_mode=RunMode.OPEN_LOOP,
        termination_rules=[{"when": "(done)", "outcome": "success", "reward": 1.0}],
        static_facts={("done", ())},
    )
    env.reset()
    obs, r, done, info = env.step("(noop)")
    assert done
    assert info["outcome"] == "success"
    # open_loop_end shouldn't override an explicit success
    assert info.get("reason") in (None, "term", "done", "success")

def test_invalid_retries_accumulate_until_valid(make_env):
    from sere.core.pddl_env.env import RunMode
    ok = ActionSpec(name="ok", params=[], pre=[], add=[], delete=[], nl=["ok"])
    env = make_env(actions={"ok": ok}, run_mode=RunMode.INTERACTIVE, illegal_move_retries=2, invalid_retry_penalty=-0.05)
    env.reset()
    # two invalids, then a valid
    env.step("(unknown)")
    env.step("(unknown)")
    obs, r, done, info = env.step("(ok)")
    assert not done
    # retries should reset after a valid action
    obs2, r2, done2, info2 = env.step("(unknown)")
    assert not done2
    assert r2 == -0.05  # back to first retry penalty

def test_obs_footer_changes_with_mode(make_env):
    from sere.core.pddl_env.env import RunMode
    a = ActionSpec(name="noop", params=[], pre=[], add=[], delete=[], nl=["noop"])

    env_i = make_env(actions={"noop": a}, run_mode=RunMode.INTERACTIVE)
    obs_i, _ = env_i.reset()
    assert obs_i.strip().endswith("Reply with (action args).")

    env_b = make_env(actions={"noop": a}, run_mode=RunMode.BATCH)
    obs_b, _ = env_b.reset()
    assert "You may submit multiple actions:" in obs_b

    env_o = make_env(actions={"noop": a}, run_mode=RunMode.OPEN_LOOP)
    obs_o, _ = env_o.reset()
    assert "Episode will end after execution." in obs_o
