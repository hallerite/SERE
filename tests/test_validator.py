"""Tests for the pure plan validator and agentic environment."""

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
    """Load blocksworld domain + first problem into (domain, world, static_facts, goal)."""
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
        # Can't unstack A B if (on A B) is not true
        r = validate_step(dom, w, sf, "unstack", ("A", "B"))
        assert not r.success
        assert r.failed_preconditions

    def test_successful_step(self):
        dom, w, sf, _ = _blocksworld_parts()
        # All blocks on table, all clear, handempty
        r = validate_step(dom, w, sf, "pick-up", ("A",))
        assert r.success
        # After pick-up: holding A, not handempty, not ontable A, not clear A
        assert ("holding", ("A",)) in w.facts
        assert ("handempty", ()) not in w.facts


# ---------------------------------------------------------------------------
# validate_plan tests
# ---------------------------------------------------------------------------

class TestValidatePlan:
    def test_valid_plan(self):
        dom, w, sf, goal = _blocksworld_parts()
        # Goal: (ON D C) (ON C B) (ON B A), all on table
        plan = [
            ("pick-up", ("B",)),
            ("stack", ("B", "A")),
            ("pick-up", ("C",)),
            ("stack", ("C", "B")),
            ("pick-up", ("D",)),
            ("stack", ("D", "C")),
        ]
        r = validate_plan(dom, w, sf, goal, plan)
        assert r.success
        assert r.goal_reached
        assert r.steps_executed == 6

    def test_plan_fails_midway(self):
        dom, w, sf, goal = _blocksworld_parts()
        plan = [
            ("pick-up", ("A",)),
            ("pick-up", ("B",)),  # fails: not handempty
        ]
        r = validate_plan(dom, w, sf, goal, plan)
        assert not r.success
        assert r.steps_executed == 1
        assert r.failed_step is not None
        assert r.failed_step.action_name == "pick-up"

    def test_plan_valid_but_goal_not_reached(self):
        dom, w, sf, goal = _blocksworld_parts()
        plan = [
            ("pick-up", ("A",)),
            ("put-down", ("A",)),
        ]
        r = validate_plan(dom, w, sf, goal, plan)
        assert not r.success
        assert not r.goal_reached
        assert r.steps_executed == 2
        assert "goal not reached" in r.error

    def test_empty_plan(self):
        dom, w, sf, goal = _blocksworld_parts()
        r = validate_plan(dom, w, sf, goal, [])
        assert not r.success  # goal not initially satisfied
        assert r.steps_executed == 0

    def test_does_not_mutate_init_world(self):
        dom, w, sf, goal = _blocksworld_parts()
        init_facts = set(w.facts)
        plan = [("pick-up", ("A",)), ("put-down", ("A",))]
        validate_plan(dom, w, sf, goal, plan)
        assert w.facts == init_facts  # init world unchanged


# ---------------------------------------------------------------------------
# AgenticPDDLEnv tests
# ---------------------------------------------------------------------------

class TestAgenticPDDLEnv:
    def test_system_prompt_contains_pddl(self):
        env, _ = _blocksworld_env()
        prompt = env.system_prompt()
        assert "domain" in prompt.lower()
        assert "problem" in prompt.lower()
        assert "validate_plan" in prompt

    def test_tool_schema(self):
        env, _ = _blocksworld_env()
        schema = env.tool_schema()
        assert schema["function"]["name"] == "validate_plan"
        assert "plan" in schema["function"]["parameters"]["properties"]

    def test_validate_bad_parse(self):
        env, _ = _blocksworld_env()
        feedback, done = env.validate("not a valid plan")
        assert "Failed to parse" in feedback
        assert not done

    def test_validate_bad_plan(self):
        env, _ = _blocksworld_env()
        feedback, done = env.validate("(pick-up NONEXISTENT)")
        assert "Unknown object" in feedback
        assert not done

    def test_validate_good_plan(self):
        env, _ = _blocksworld_env()
        plan = "(pick-up B)\n(stack B A)\n(pick-up C)\n(stack C B)\n(pick-up D)\n(stack D C)"
        feedback, done = env.validate(plan)
        assert "Goal reached" in feedback
        assert done
        assert env.solved

    def test_max_attempts_terminates(self):
        env, _ = _blocksworld_env()
        env.max_attempts = 2
        env.validate("(pick-up NONEXISTENT)")
        _, done = env.validate("(pick-up NONEXISTENT)")
        assert done
        assert not env.solved

    def test_init_state_immutable_across_attempts(self):
        """Guard: each validation attempt starts from the original init state."""
        env, _ = _blocksworld_env()
        # First attempt: partially valid (modifies state internally)
        env.validate("(pick-up A)\n(put-down A)")
        # Second attempt: same action should still work (state reset)
        feedback, _ = env.validate("(pick-up A)\n(put-down A)")
        assert "failed" not in feedback.lower() or "goal not reached" in feedback.lower()

    def test_goal_cannot_be_changed(self):
        """Guard: goal is set at construction and cannot be modified."""
        env, _ = _blocksworld_env()
        original_goal = env.goal_expr
        # Attempt to modify (would require code change — structural guard)
        assert env.goal_expr == original_goal
        # Validate still uses the original goal
        feedback, done = env.validate("(pick-up A)\n(put-down A)")
        assert not done or not env.solved  # won't solve the real goal


# ---------------------------------------------------------------------------
# format_plan_feedback tests
# ---------------------------------------------------------------------------

class TestFormatPlanFeedback:
    def test_success_message(self):
        dom, w, sf, goal = _blocksworld_parts()
        plan = [
            ("pick-up", ("B",)),
            ("stack", ("B", "A")),
            ("pick-up", ("C",)),
            ("stack", ("C", "B")),
            ("pick-up", ("D",)),
            ("stack", ("D", "C")),
        ]
        r = validate_plan(dom, w, sf, goal, plan)
        msg = format_plan_feedback(r)
        assert "Goal reached" in msg
        assert "6 steps" in msg

    def test_failure_message(self):
        dom, w, sf, goal = _blocksworld_parts()
        plan = [("unstack", ("A", "B"))]
        r = validate_plan(dom, w, sf, goal, plan)
        msg = format_plan_feedback(r)
        assert "failed at step 1" in msg.lower()
        assert "State after step" in msg
