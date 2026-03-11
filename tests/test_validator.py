"""Tests for the plan validator and miniSWE-style agentic environment."""

import pytest
from importlib.resources import files as pkg_files

from sere.core.validator import validate_plan, validate_step, check_goal, format_plan_feedback
from sere.core.agentic_env import AgenticPDDLEnv
from sere.io.pddl_loader import load_agentic_task, load_pddl_domain
from sere.pddl.pddl_parser import parse_problem_file
from sere.core.world_state import WorldState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blocksworld_env():
    """Load the first blocksworld problem as an agentic env."""
    bw = pkg_files("sere.assets.pddl") / "blocksworld"
    prob = sorted((bw / "problems").iterdir())[0]
    return load_agentic_task(str(bw), str(prob))


def _blocksworld_parts():
    """Load blocksworld domain + first problem."""
    bw = pkg_files("sere.assets.pddl") / "blocksworld"
    dom, _meta, pddl_domain = load_pddl_domain(str(bw))
    prob_path = sorted((bw / "problems").iterdir())[0]
    problem = parse_problem_file(prob_path)

    w = WorldState(dom)
    for const_name, const_type in pddl_domain.constants:
        w.add_object(const_name, const_type)
    for obj_name, obj_type in problem.objects:
        w.add_object(obj_name, obj_type)

    static_preds = {name for name, spec in dom.predicates.items() if spec.static}
    static_facts = set()
    for pred, args in problem.init_facts:
        if pred in static_preds:
            static_facts.add((pred, args))
        else:
            w.facts.add((pred, args))

    return dom, w, static_facts, problem.goal


VALID_PLAN = "(pick-up B)\n(stack B A)\n(pick-up C)\n(stack C B)\n(pick-up D)\n(stack D C)"


# ---------------------------------------------------------------------------
# validate_step tests
# ---------------------------------------------------------------------------

class TestValidateStep:
    def test_unknown_action(self):
        dom, w, sf, _ = _blocksworld_parts()
        r = validate_step(dom, w, sf, "nonexistent", ("A",))
        assert not r.success
        assert "Unknown action" in r.error

    def test_arity_mismatch(self):
        dom, w, sf, _ = _blocksworld_parts()
        r = validate_step(dom, w, sf, "pick-up", ())
        assert not r.success
        assert "Arity mismatch" in r.error

    def test_unknown_object(self):
        dom, w, sf, _ = _blocksworld_parts()
        r = validate_step(dom, w, sf, "pick-up", ("NONEXISTENT",))
        assert not r.success
        assert "Unknown object" in r.error

    def test_precondition_failure(self):
        dom, w, sf, _ = _blocksworld_parts()
        r = validate_step(dom, w, sf, "unstack", ("A", "B"))
        assert not r.success
        assert r.failed_preconditions

    def test_successful_step(self):
        dom, w, sf, _ = _blocksworld_parts()
        r = validate_step(dom, w, sf, "pick-up", ("A",))
        assert r.success
        assert ("holding", ("A",)) in w.facts
        assert ("handempty", ()) not in w.facts


# ---------------------------------------------------------------------------
# validate_plan tests
# ---------------------------------------------------------------------------

class TestValidatePlan:
    def test_valid_plan(self):
        dom, w, sf, goal = _blocksworld_parts()
        plan = [
            ("pick-up", ("B",)), ("stack", ("B", "A")),
            ("pick-up", ("C",)), ("stack", ("C", "B")),
            ("pick-up", ("D",)), ("stack", ("D", "C")),
        ]
        r = validate_plan(dom, w, sf, goal, plan)
        assert r.success
        assert r.goal_reached
        assert r.steps_executed == 6

    def test_plan_fails_midway(self):
        dom, w, sf, goal = _blocksworld_parts()
        plan = [("pick-up", ("A",)), ("pick-up", ("B",))]
        r = validate_plan(dom, w, sf, goal, plan)
        assert not r.success
        assert r.steps_executed == 1
        assert r.failed_step.action_name == "pick-up"

    def test_plan_valid_but_goal_not_reached(self):
        dom, w, sf, goal = _blocksworld_parts()
        plan = [("pick-up", ("A",)), ("put-down", ("A",))]
        r = validate_plan(dom, w, sf, goal, plan)
        assert not r.success
        assert not r.goal_reached
        assert "goal not reached" in r.error

    def test_empty_plan(self):
        dom, w, sf, goal = _blocksworld_parts()
        r = validate_plan(dom, w, sf, goal, [])
        assert not r.success
        assert r.steps_executed == 0

    def test_does_not_mutate_init_world(self):
        dom, w, sf, goal = _blocksworld_parts()
        init_facts = set(w.facts)
        validate_plan(dom, w, sf, goal, [("pick-up", ("A",)), ("put-down", ("A",))])
        assert w.facts == init_facts


# ---------------------------------------------------------------------------
# AgenticPDDLEnv — file tools
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_read_domain(self):
        env, _ = _blocksworld_env()
        result, done = env.handle_tool_call("read_file", {"path": "domain.pddl"})
        assert "define" in result.lower()
        assert not done

    def test_read_problem(self):
        env, _ = _blocksworld_env()
        result, done = env.handle_tool_call("read_file", {"path": "problem.pddl"})
        assert ":goal" in result.lower()
        assert not done

    def test_read_empty_plan(self):
        env, _ = _blocksworld_env()
        result, done = env.handle_tool_call("read_file", {"path": "plan.pddl"})
        assert "empty" in result.lower()
        assert not done

    def test_read_plan_after_write(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        result, _ = env.handle_tool_call("read_file", {"path": "plan.pddl"})
        assert "(pick-up B)" in result

    def test_read_unknown_file(self):
        env, _ = _blocksworld_env()
        result, _ = env.handle_tool_call("read_file", {"path": "foo.txt"})
        assert "not found" in result.lower()


class TestWriteFile:
    def test_write_plan(self):
        env, _ = _blocksworld_env()
        result, done = env.handle_tool_call("write_file", {"content": VALID_PLAN})
        assert "6 actions" in result
        assert not done

    def test_write_empty_rejected(self):
        env, _ = _blocksworld_env()
        result, _ = env.handle_tool_call("write_file", {"content": ""})
        assert "empty" in result.lower()

    def test_write_bad_syntax_warns(self):
        env, _ = _blocksworld_env()
        result, _ = env.handle_tool_call("write_file", {"content": "not valid pddl"})
        assert "warning" in result.lower()

    def test_write_does_not_count_as_attempt(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        assert env.attempts == 0


# ---------------------------------------------------------------------------
# AgenticPDDLEnv — validate tool
# ---------------------------------------------------------------------------

class TestValidateTool:
    def test_validate_no_plan(self):
        env, _ = _blocksworld_env()
        result, done = env.handle_tool_call("validate", {})
        assert "empty" in result.lower()
        assert not done

    def test_validate_good_plan(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        result, done = env.handle_tool_call("validate", {})
        assert "Goal reached" in result
        assert done
        assert env.solved
        assert env.attempts == 1

    def test_validate_bad_plan(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": "(pick-up NONEXISTENT)"})
        result, done = env.handle_tool_call("validate", {})
        assert "Unknown object" in result
        assert not done
        assert env.attempts == 1

    def test_validate_partial_does_not_count(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        result, done = env.handle_tool_call("validate", {"up_to_step": 2})
        assert not done
        assert env.attempts == 0  # partial doesn't count

    def test_validate_partial_shows_subplan(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        result, done = env.handle_tool_call("validate", {"up_to_step": 2})
        # First 2 steps are valid but don't reach goal
        assert "goal not reached" in result.lower()

    def test_max_attempts_terminates(self):
        env, _ = _blocksworld_env()
        env.max_attempts = 2
        env.handle_tool_call("write_file", {"content": "(pick-up NONEXISTENT)"})
        env.handle_tool_call("validate", {})
        _, done = env.handle_tool_call("validate", {})
        assert done
        assert not env.solved


# ---------------------------------------------------------------------------
# AgenticPDDLEnv — simulate tool
# ---------------------------------------------------------------------------

class TestSimulateTool:
    def test_simulate_full_plan(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        result, done = env.handle_tool_call("simulate", {})
        assert "Goal satisfied: True" in result
        assert not done  # simulate never ends episode

    def test_simulate_partial(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        result, _ = env.handle_tool_call("simulate", {"up_to_step": 1})
        assert "(holding B)" in result.lower() or "(holding b)" in result.lower()

    def test_simulate_no_plan(self):
        env, _ = _blocksworld_env()
        result, _ = env.handle_tool_call("simulate", {})
        assert "empty" in result.lower()

    def test_simulate_does_not_mutate_state(self):
        env, _ = _blocksworld_env()
        init_facts = set(env.init_world.facts)
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        env.handle_tool_call("simulate", {})
        assert env.init_world.facts == init_facts

    def test_simulate_does_not_count_as_attempt(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        env.handle_tool_call("simulate", {})
        assert env.attempts == 0


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards:
    def test_init_state_immutable_across_validates(self):
        env, _ = _blocksworld_env()
        env.handle_tool_call("write_file", {"content": "(pick-up A)\n(put-down A)"})
        env.handle_tool_call("validate", {})
        # Second attempt should still work (state reset)
        env.handle_tool_call("write_file", {"content": "(pick-up A)\n(put-down A)"})
        result, _ = env.handle_tool_call("validate", {})
        assert "goal not reached" in result.lower()

    def test_domain_file_immutable(self):
        env, _ = _blocksworld_env()
        original = env.domain_pddl
        # Only write_file writes plan.pddl — no way to modify domain
        env.handle_tool_call("write_file", {"content": "hacked domain"})
        assert env.domain_pddl == original

    def test_problem_file_immutable(self):
        env, _ = _blocksworld_env()
        original = env.problem_pddl
        env.handle_tool_call("write_file", {"content": "hacked problem"})
        assert env.problem_pddl == original

    def test_goal_immutable(self):
        env, _ = _blocksworld_env()
        original = env.goal_expr
        env.handle_tool_call("write_file", {"content": VALID_PLAN})
        env.handle_tool_call("validate", {})
        assert env.goal_expr == original


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_unknown_tool(self):
        env, _ = _blocksworld_env()
        result, done = env.handle_tool_call("hack_the_planet", {})
        assert "Unknown tool" in result
        assert not done


# ---------------------------------------------------------------------------
# format_plan_feedback
# ---------------------------------------------------------------------------

class TestFormatPlanFeedback:
    def test_success_message(self):
        dom, w, sf, goal = _blocksworld_parts()
        plan = [
            ("pick-up", ("B",)), ("stack", ("B", "A")),
            ("pick-up", ("C",)), ("stack", ("C", "B")),
            ("pick-up", ("D",)), ("stack", ("D", "C")),
        ]
        msg = format_plan_feedback(validate_plan(dom, w, sf, goal, plan))
        assert "Goal reached" in msg
        assert "6 steps" in msg

    def test_failure_message(self):
        dom, w, sf, goal = _blocksworld_parts()
        msg = format_plan_feedback(validate_plan(dom, w, sf, goal, [("unstack", ("A", "B"))]))
        assert "failed at step 1" in msg.lower()
        assert "State after step" in msg
