"""
Agentic PDDL environment — miniSWE-style.

The LLM gets domain.pddl + problem.pddl and a set of tools:
  - validate_plan: submit a complete plan, get pass/fail + diagnostics
  - apply_and_show: apply a partial prefix, see the resulting state
  - check_action: check if a single action is valid in the current state

The agent drives its own workflow: inspect, experiment, then submit.
Guards ensure the goal and initial state cannot be changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from sere.pddl.domain_spec import DomainSpec, Predicate
from sere.core.world_state import WorldState
from sere.core.validator import (
    PlanResult,
    StepResult,
    check_goal,
    format_plan_feedback,
    format_step_error,
    validate_plan,
    validate_step,
)
from sere.core.pddl_env.planning import parse_actions


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "validate_plan",
            "description": (
                "Submit a complete plan for validation. The plan is simulated "
                "from the initial state. Returns success or the first failing "
                "step with diagnostics. This is the only way to solve the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": (
                            "Sequence of grounded PDDL actions, one per line:\n"
                            "(action-name arg1 arg2)\n(action-name arg1 ...)"
                        ),
                    },
                },
                "required": ["plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_prefix",
            "description": (
                "Apply a prefix of actions from the initial state and show "
                "the resulting state. Use this to explore intermediate states "
                "and debug your plan. Does NOT count as a submission."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "string",
                        "description": (
                            "Sequence of grounded PDDL actions to apply:\n"
                            "(action-name arg1 arg2)\n(action-name arg1 ...)"
                        ),
                    },
                },
                "required": ["actions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_action",
            "description": (
                "Check if a single action is valid in the initial state "
                "(or after applying an optional prefix). Returns whether "
                "preconditions are satisfied and why/why not."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "A single grounded PDDL action: (action-name arg1 arg2)",
                    },
                    "after": {
                        "type": "string",
                        "description": (
                            "Optional prefix of actions to apply first. "
                            "If omitted, checks against the initial state."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    },
]


@dataclass
class AgenticPDDLEnv:
    """
    miniSWE-style PDDL planning environment.

    The agent has tools to explore, debug, and submit plans.
    State resets from scratch for each tool call — no carryover.
    """

    domain: DomainSpec
    init_world: WorldState
    static_facts: Set[Predicate]
    goal_expr: str
    domain_pddl: str
    problem_pddl: str
    problem_name: str = ""

    # Feature flags
    enable_numeric: bool = False
    enable_conditional: bool = False

    # Limits
    max_attempts: int = 8

    # State
    attempts: int = 0
    solved: bool = False

    def system_prompt(self) -> str:
        """Build the system prompt with domain + problem PDDL."""
        return (
            "You are a PDDL planning agent. Given a planning domain and problem, "
            "write a complete plan that achieves the goal.\n\n"
            "You have tools to help:\n"
            "- `validate_plan`: submit your complete plan for grading\n"
            "- `apply_prefix`: apply a partial plan and see the resulting state\n"
            "- `check_action`: check if a specific action is valid in a given state\n\n"
            "Use apply_prefix and check_action to explore and debug. "
            "When ready, submit with validate_plan.\n\n"
            f"=== DOMAIN ===\n{self.domain_pddl}\n\n"
            f"=== PROBLEM ===\n{self.problem_pddl}"
        )

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool definitions."""
        return list(TOOL_SCHEMAS)

    # -- legacy single-schema accessor --
    def tool_schema(self) -> Dict[str, Any]:
        return TOOL_SCHEMAS[0]

    def _fresh_world(self) -> WorldState:
        """Deep copy of init state."""
        return WorldState(
            domain=self.init_world.domain,
            objects={k: set(v) for k, v in self.init_world.objects.items()},
            facts=set(self.init_world.facts),
            fluents=dict(self.init_world.fluents),
        )

    # -----------------------------------------------------------------
    #  Tool handlers
    # -----------------------------------------------------------------

    def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Dispatch a tool call. Returns (result_text, is_done).
        Only validate_plan can end the episode.
        """
        if name == "validate_plan":
            return self.validate(arguments.get("plan", ""))
        elif name == "apply_prefix":
            return self.apply_prefix(arguments.get("actions", "")), False
        elif name == "check_action":
            return self.check_action(
                arguments.get("action", ""),
                arguments.get("after"),
            ), False
        else:
            return f"Unknown tool: {name}", False

    def validate(self, plan_text: str) -> Tuple[str, bool]:
        """Submit a plan for validation. Counts as an attempt."""
        self.attempts += 1

        try:
            plan = parse_actions(plan_text)
        except ValueError as e:
            feedback = (
                f"Failed to parse plan: {e}\n\n"
                "Submit actions as S-expressions, one per line:\n"
                "  (action-name arg1 arg2)\n"
                "  (action-name arg1 arg2)"
            )
            done = self.attempts >= self.max_attempts
            if done:
                feedback += f"\n\nMax attempts ({self.max_attempts}) reached."
            return feedback, done

        result = validate_plan(
            self.domain,
            self.init_world,
            self.static_facts,
            self.goal_expr,
            plan,
            enable_numeric=self.enable_numeric,
            enable_conditional=self.enable_conditional,
        )

        self.solved = result.success
        feedback = format_plan_feedback(result)

        done = result.success or self.attempts >= self.max_attempts
        if not result.success and self.attempts >= self.max_attempts:
            feedback += f"\n\nMax attempts ({self.max_attempts}) reached."

        return feedback, done

    def apply_prefix(self, actions_text: str) -> str:
        """Apply a prefix of actions and return the resulting state."""
        try:
            actions = parse_actions(actions_text)
        except ValueError as e:
            return f"Failed to parse actions: {e}"

        world = self._fresh_world()

        for i, (name, args) in enumerate(actions):
            result = validate_step(
                self.domain, world, self.static_facts, name, args,
                enable_numeric=self.enable_numeric,
                enable_conditional=self.enable_conditional,
            )
            if not result.success:
                return format_step_error(result, i) + "\n\n" + _format_state(world)

        # Show resulting state
        goal_met = check_goal(world, self.static_facts, self.goal_expr, self.enable_numeric)
        lines = [f"After {len(actions)} actions:"]
        lines.append(_format_state(world))
        lines.append(f"Goal satisfied: {goal_met}")
        return "\n".join(lines)

    def check_action(self, action_text: str, after: Optional[str] = None) -> str:
        """Check if a single action is valid, optionally after a prefix."""
        try:
            action_list = parse_actions(action_text)
            if len(action_list) != 1:
                return "Please provide exactly one action to check."
            action_name, action_args = action_list[0]
        except ValueError as e:
            return f"Failed to parse action: {e}"

        world = self._fresh_world()

        # Apply optional prefix
        if after:
            try:
                prefix = parse_actions(after)
            except ValueError as e:
                return f"Failed to parse prefix: {e}"
            for i, (name, args) in enumerate(prefix):
                result = validate_step(
                    self.domain, world, self.static_facts, name, args,
                    enable_numeric=self.enable_numeric,
                    enable_conditional=self.enable_conditional,
                )
                if not result.success:
                    return f"Prefix failed at step {i + 1}: {result.error}"

        # Check the action
        from sere.core.validator import _resolve_action
        from sere.core.semantics import eval_trace

        act, bind, err = _resolve_action(self.domain, world, action_name, action_args)
        if err:
            return f"Invalid: {err}"

        assert act is not None and bind is not None

        lines = [f"Checking: ({action_name} {' '.join(action_args)})"]
        all_ok = True
        for pre in act.pre or []:
            node = eval_trace(
                world, self.static_facts, pre, bind,
                enable_numeric=self.enable_numeric,
            )
            status = "OK" if node.satisfied else "FAIL"
            if not node.satisfied:
                all_ok = False
            lines.append(f"  [{status}] {node.expr}")

        lines.append(f"Result: {'valid' if all_ok else 'preconditions not met'}")
        return "\n".join(lines)


def _format_state(world: WorldState) -> str:
    """Format world state as PDDL-style facts."""
    lines = ["State:"]
    for pred, args in sorted(world.facts):
        lines.append(f"  ({pred} {' '.join(args)})")
    for (fname, fargs), fval in sorted(world.fluents.items()):
        lines.append(f"  (= ({fname} {' '.join(fargs)}) {fval})")
    return "\n".join(lines)
