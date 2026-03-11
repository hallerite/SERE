"""Tests for the plan validator and miniSWE-style agentic sandbox."""

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

@pytest.fixture
def bw_env():
    """Load the first blocksworld problem as an agentic env, cleanup after."""
    bw = pkg_files("sere.assets.pddl") / "blocksworld"
    prob = sorted((bw / "problems").iterdir())[0]
    env, meta = load_agentic_task(str(bw), str(prob))
    yield env
    env.cleanup()


def _blocksworld_parts():
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
# validate_step / validate_plan (pure validator)
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


class TestValidatePlan:
    def test_valid_plan(self):
        dom, w, sf, goal = _blocksworld_parts()
        plan = [
            ("pick-up", ("B",)), ("stack", ("B", "A")),
            ("pick-up", ("C",)), ("stack", ("C", "B")),
            ("pick-up", ("D",)), ("stack", ("D", "C")),
        ]
        r = validate_plan(dom, w, sf, goal, plan)
        assert r.success and r.goal_reached and r.steps_executed == 6

    def test_plan_fails_midway(self):
        dom, w, sf, goal = _blocksworld_parts()
        r = validate_plan(dom, w, sf, goal, [("pick-up", ("A",)), ("pick-up", ("B",))])
        assert not r.success
        assert r.steps_executed == 1

    def test_goal_not_reached(self):
        dom, w, sf, goal = _blocksworld_parts()
        r = validate_plan(dom, w, sf, goal, [("pick-up", ("A",)), ("put-down", ("A",))])
        assert not r.success and "goal not reached" in r.error

    def test_empty_plan(self):
        dom, w, sf, goal = _blocksworld_parts()
        r = validate_plan(dom, w, sf, goal, [])
        assert not r.success

    def test_does_not_mutate_init_world(self):
        dom, w, sf, goal = _blocksworld_parts()
        init_facts = set(w.facts)
        validate_plan(dom, w, sf, goal, [("pick-up", ("A",)), ("put-down", ("A",))])
        assert w.facts == init_facts


# ---------------------------------------------------------------------------
# Sandbox workspace
# ---------------------------------------------------------------------------

class TestWorkspace:
    def test_workspace_created_lazily(self, bw_env):
        # Workspace shouldn't exist until first access
        assert bw_env._workspace is None
        ws = bw_env.workspace
        assert ws.exists()
        assert (ws / "domain.pddl").exists()
        assert (ws / "problem.pddl").exists()

    def test_cleanup_removes_workspace(self, bw_env):
        ws = bw_env.workspace
        assert ws.exists()
        bw_env.cleanup()
        assert not ws.exists()


# ---------------------------------------------------------------------------
# bash tool
# ---------------------------------------------------------------------------

class TestBash:
    def test_ls(self, bw_env):
        result, done = bw_env.handle_tool_call("bash", {"command": "ls"})
        assert "domain.pddl" in result
        assert "problem.pddl" in result
        assert not done

    def test_cat_domain(self, bw_env):
        result, _ = bw_env.handle_tool_call("bash", {"command": "cat domain.pddl"})
        assert "define" in result.lower()

    def test_grep(self, bw_env):
        result, _ = bw_env.handle_tool_call("bash", {"command": "grep -c ':action' domain.pddl"})
        assert int(result.strip()) > 0

    def test_write_plan_via_bash(self, bw_env):
        bw_env.handle_tool_call("bash", {"command": f"cat > plan.pddl << 'EOF'\n{VALID_PLAN}\nEOF"})
        result, done = bw_env.handle_tool_call("validate", {})
        assert "Goal reached" in result and done

    def test_bash_readonly_guard(self, bw_env):
        """Even if bash modifies domain.pddl, it gets restored."""
        bw_env.handle_tool_call("bash", {"command": "echo 'hacked' > domain.pddl"})
        result, _ = bw_env.handle_tool_call("read_file", {"path": "domain.pddl"})
        assert "hacked" not in result
        assert "define" in result.lower()

    def test_bash_empty_command(self, bw_env):
        result, _ = bw_env.handle_tool_call("bash", {"command": ""})
        assert "error" in result.lower()

    def test_bash_wc(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": VALID_PLAN})
        result, _ = bw_env.handle_tool_call("bash", {"command": "grep -c '(' plan.pddl"})
        assert "6" in result


# ---------------------------------------------------------------------------
# read_file tool
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_read_domain(self, bw_env):
        result, _ = bw_env.handle_tool_call("read_file", {"path": "domain.pddl"})
        assert "define" in result.lower()

    def test_read_problem(self, bw_env):
        result, _ = bw_env.handle_tool_call("read_file", {"path": "problem.pddl"})
        assert ":goal" in result.lower()

    def test_read_nonexistent(self, bw_env):
        result, _ = bw_env.handle_tool_call("read_file", {"path": "foo.txt"})
        assert "not found" in result.lower()

    def test_read_plan_after_write(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": VALID_PLAN})
        result, _ = bw_env.handle_tool_call("read_file", {"path": "plan.pddl"})
        assert "(pick-up B)" in result


# ---------------------------------------------------------------------------
# write_file tool
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_write_plan(self, bw_env):
        result, done = bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": VALID_PLAN})
        assert "6 actions" in result
        assert not done

    def test_write_readonly_rejected(self, bw_env):
        result, _ = bw_env.handle_tool_call("write_file", {"path": "domain.pddl", "content": "hack"})
        assert "read-only" in result.lower()

    def test_write_scratch_file(self, bw_env):
        result, _ = bw_env.handle_tool_call("write_file", {"path": "notes.txt", "content": "hello"})
        assert "Wrote" in result
        content, _ = bw_env.handle_tool_call("read_file", {"path": "notes.txt"})
        assert content == "hello"


# ---------------------------------------------------------------------------
# str_replace tool
# ---------------------------------------------------------------------------

class TestStrReplace:
    def test_replace_in_plan(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": "(pick-up A)\n(put-down A)"})
        result, _ = bw_env.handle_tool_call("str_replace", {
            "path": "plan.pddl",
            "old_str": "(put-down A)",
            "new_str": "(stack A B)",
        })
        assert "Replaced" in result
        content, _ = bw_env.handle_tool_call("read_file", {"path": "plan.pddl"})
        assert "(stack A B)" in content
        assert "(put-down A)" not in content

    def test_replace_not_found(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": "(pick-up A)"})
        result, _ = bw_env.handle_tool_call("str_replace", {
            "path": "plan.pddl",
            "old_str": "nonexistent",
            "new_str": "whatever",
        })
        assert "not found" in result.lower()

    def test_replace_ambiguous(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": "(pick-up A)\n(pick-up A)"})
        result, _ = bw_env.handle_tool_call("str_replace", {
            "path": "plan.pddl",
            "old_str": "(pick-up A)",
            "new_str": "(pick-up B)",
        })
        assert "2 times" in result

    def test_replace_readonly_rejected(self, bw_env):
        result, _ = bw_env.handle_tool_call("str_replace", {
            "path": "domain.pddl",
            "old_str": "define",
            "new_str": "hack",
        })
        assert "read-only" in result.lower()


# ---------------------------------------------------------------------------
# validate tool
# ---------------------------------------------------------------------------

class TestValidateTool:
    def test_validate_empty(self, bw_env):
        result, done = bw_env.handle_tool_call("validate", {})
        assert "empty" in result.lower()
        assert not done

    def test_validate_good_plan(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": VALID_PLAN})
        result, done = bw_env.handle_tool_call("validate", {})
        assert "Goal reached" in result and done and bw_env.solved
        assert bw_env.attempts == 1

    def test_validate_bad_plan(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": "(pick-up NONEXISTENT)"})
        result, done = bw_env.handle_tool_call("validate", {})
        assert "Unknown object" in result
        assert not done and bw_env.attempts == 1

    def test_validate_partial_free(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": VALID_PLAN})
        result, done = bw_env.handle_tool_call("validate", {"up_to_step": 2})
        assert not done and bw_env.attempts == 0

    def test_max_attempts(self, bw_env):
        bw_env.max_attempts = 2
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": "(pick-up NONEXISTENT)"})
        bw_env.handle_tool_call("validate", {})
        _, done = bw_env.handle_tool_call("validate", {})
        assert done and not bw_env.solved


# ---------------------------------------------------------------------------
# simulate tool
# ---------------------------------------------------------------------------

class TestSimulateTool:
    def test_simulate_full(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": VALID_PLAN})
        result, done = bw_env.handle_tool_call("simulate", {})
        assert "Goal satisfied: True" in result
        assert not done

    def test_simulate_partial(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": VALID_PLAN})
        result, _ = bw_env.handle_tool_call("simulate", {"up_to_step": 1})
        assert "holding" in result.lower()

    def test_simulate_empty(self, bw_env):
        result, _ = bw_env.handle_tool_call("simulate", {})
        assert "empty" in result.lower()

    def test_simulate_does_not_count(self, bw_env):
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": VALID_PLAN})
        bw_env.handle_tool_call("simulate", {})
        assert bw_env.attempts == 0


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards:
    def test_init_state_immutable(self, bw_env):
        init_facts = set(bw_env.init_world.facts)
        bw_env.handle_tool_call("write_file", {"path": "plan.pddl", "content": VALID_PLAN})
        bw_env.handle_tool_call("simulate", {})
        bw_env.handle_tool_call("validate", {})
        assert bw_env.init_world.facts == init_facts

    def test_domain_restored_after_bash(self, bw_env):
        original = bw_env.domain_pddl
        bw_env.handle_tool_call("bash", {"command": "rm domain.pddl"})
        assert (bw_env.workspace / "domain.pddl").read_text() == original

    def test_problem_restored_after_bash(self, bw_env):
        original = bw_env.problem_pddl
        bw_env.handle_tool_call("bash", {"command": "echo 'x' > problem.pddl"})
        assert (bw_env.workspace / "problem.pddl").read_text() == original


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
        assert "Goal reached" in msg and "6 steps" in msg

    def test_failure_message(self):
        dom, w, sf, goal = _blocksworld_parts()
        msg = format_plan_feedback(validate_plan(dom, w, sf, goal, [("unstack", ("A", "B"))]))
        assert "failed at step 1" in msg.lower()
