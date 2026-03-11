"""
miniSWE-style PDDL planning environment.

The agent works in a virtual workspace with files:
  - domain.pddl   (read-only)
  - problem.pddl   (read-only)
  - plan.pddl      (read-write — the agent's solution)

Tools mirror miniSWE: read_file, write_file + two PDDL-specific:
  - validate: run plan.pddl against domain+problem, pass/fail + diagnostics
  - simulate: run plan.pddl up to step N, return resulting world state

Guards: domain.pddl and problem.pddl are immutable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from sere.pddl.domain_spec import DomainSpec, Predicate
from sere.core.world_state import WorldState
from sere.core.validator import (
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
            "name": "read_file",
            "description": "Read a file from the workspace. Available files: domain.pddl, problem.pddl, plan.pddl",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read (domain.pddl, problem.pddl, or plan.pddl)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to plan.pddl. This is the only writable file. "
                "The plan should be a sequence of grounded actions, one per line:\n"
                "  (action-name arg1 arg2)\n  (action-name arg1 arg2)\n  ..."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The plan content to write",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate",
            "description": (
                "Validate plan.pddl against the domain and problem. "
                "Returns success or the first failing step with diagnostics. "
                "Optionally validate only up to a given step number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "up_to_step": {
                        "type": "integer",
                        "description": "Only validate the first N steps of the plan. If omitted, validates the entire plan.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate",
            "description": (
                "Run plan.pddl up to a given step and return the resulting "
                "world state. Use this to inspect intermediate states."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "up_to_step": {
                        "type": "integer",
                        "description": "Run the first N steps and show the state. If omitted, runs the entire plan.",
                    },
                },
            },
        },
    },
]


@dataclass
class AgenticPDDLEnv:
    """
    miniSWE-style PDDL planning environment.

    Virtual workspace with domain.pddl (ro), problem.pddl (ro), plan.pddl (rw).
    """

    domain: DomainSpec
    init_world: WorldState
    static_facts: Set[Predicate]
    goal_expr: str
    domain_pddl: str
    problem_pddl: str
    problem_name: str = ""

    enable_numeric: bool = False
    enable_conditional: bool = False

    max_attempts: int = 8

    # Mutable state
    plan_pddl: str = ""
    attempts: int = 0
    solved: bool = False

    def system_prompt(self) -> str:
        return (
            "You are a PDDL planning agent. Your workspace contains:\n"
            "  - domain.pddl   (the planning domain — read-only)\n"
            "  - problem.pddl  (the planning problem — read-only)\n"
            "  - plan.pddl     (your solution — write your plan here)\n\n"
            "Tools:\n"
            "  read_file(path)          — read any workspace file\n"
            "  write_file(content)      — write plan.pddl\n"
            "  validate(up_to_step?)    — validate plan.pddl, get pass/fail + diagnostics\n"
            "  simulate(up_to_step?)    — run plan.pddl, see resulting world state\n\n"
            "Workflow: read the domain and problem, write a plan, "
            "use simulate to inspect intermediate states, "
            "then validate to submit.\n\n"
            "The plan file should contain grounded actions, one per line:\n"
            "  (action-name arg1 arg2)\n"
            "  (action-name arg1 arg2)\n"
            "  ..."
        )

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return list(TOOL_SCHEMAS)

    # legacy
    def tool_schema(self) -> Dict[str, Any]:
        return TOOL_SCHEMAS[2]  # validate

    def _fresh_world(self) -> WorldState:
        return WorldState(
            domain=self.init_world.domain,
            objects={k: set(v) for k, v in self.init_world.objects.items()},
            facts=set(self.init_world.facts),
            fluents=dict(self.init_world.fluents),
        )

    def _parse_plan(self) -> List[Tuple[str, Tuple[str, ...]]]:
        """Parse the current plan.pddl content."""
        if not self.plan_pddl.strip():
            raise ValueError("plan.pddl is empty. Write a plan first.")
        return parse_actions(self.plan_pddl)

    # -----------------------------------------------------------------
    #  Tool dispatch
    # -----------------------------------------------------------------

    def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Dispatch a tool call. Returns (result_text, is_done).
        Only validate (full plan, no up_to_step) can end the episode.
        """
        if name == "read_file":
            return self._read_file(arguments.get("path", "")), False
        elif name == "write_file":
            return self._write_file(arguments.get("content", "")), False
        elif name == "validate":
            return self._validate(arguments.get("up_to_step"))
        elif name == "simulate":
            return self._simulate(arguments.get("up_to_step")), False
        else:
            return f"Unknown tool: {name}. Available: read_file, write_file, validate, simulate", False

    # -----------------------------------------------------------------
    #  Tool implementations
    # -----------------------------------------------------------------

    def _read_file(self, path: str) -> str:
        path = path.strip().lstrip("/")
        if path == "domain.pddl":
            return self.domain_pddl
        elif path == "problem.pddl":
            return self.problem_pddl
        elif path == "plan.pddl":
            if not self.plan_pddl:
                return "(empty — no plan written yet)"
            return self.plan_pddl
        else:
            return f"File not found: {path}. Available: domain.pddl, problem.pddl, plan.pddl"

    def _write_file(self, content: str) -> str:
        if not content.strip():
            return "Error: empty content. Write at least one action."
        self.plan_pddl = content
        # Count lines that look like actions
        try:
            actions = parse_actions(content)
            return f"Wrote plan.pddl ({len(actions)} actions)"
        except ValueError:
            self.plan_pddl = content
            return "Wrote plan.pddl (warning: could not parse as actions — check syntax)"

    def _validate(self, up_to_step: int | None = None) -> Tuple[str, bool]:
        """Validate plan.pddl. Full validation counts as an attempt."""
        try:
            plan = self._parse_plan()
        except ValueError as e:
            return str(e), False

        if up_to_step is not None:
            # Partial validation — does NOT count as attempt, never ends episode
            plan = plan[:up_to_step]
            result = validate_plan(
                self.domain, self.init_world, self.static_facts,
                self.goal_expr, plan,
                enable_numeric=self.enable_numeric,
                enable_conditional=self.enable_conditional,
            )
            feedback = format_plan_feedback(result)
            return feedback, False

        # Full validation — counts as attempt
        self.attempts += 1
        result = validate_plan(
            self.domain, self.init_world, self.static_facts,
            self.goal_expr, plan,
            enable_numeric=self.enable_numeric,
            enable_conditional=self.enable_conditional,
        )
        self.solved = result.success
        feedback = format_plan_feedback(result)

        done = result.success or self.attempts >= self.max_attempts
        if not result.success and self.attempts >= self.max_attempts:
            feedback += f"\n\nMax attempts ({self.max_attempts}) reached."

        return feedback, done

    def _simulate(self, up_to_step: int | None = None) -> str:
        """Run plan.pddl up to step N and show the world state."""
        try:
            plan = self._parse_plan()
        except ValueError as e:
            return str(e)

        if up_to_step is not None:
            plan = plan[:up_to_step]

        world = self._fresh_world()

        for i, (name, args) in enumerate(plan):
            result = validate_step(
                self.domain, world, self.static_facts, name, args,
                enable_numeric=self.enable_numeric,
                enable_conditional=self.enable_conditional,
            )
            if not result.success:
                return (
                    format_step_error(result, i) + "\n\n"
                    + _format_state(world, self.static_facts, self.goal_expr, self.enable_numeric)
                )

        return _format_state(
            world, self.static_facts, self.goal_expr, self.enable_numeric,
            prefix=f"State after {len(plan)} steps:",
        )


def _format_state(
    world: WorldState,
    static_facts: Set[Predicate] | None = None,
    goal_expr: str | None = None,
    enable_numeric: bool = False,
    prefix: str = "State:",
) -> str:
    lines = [prefix]
    for pred, args in sorted(world.facts):
        lines.append(f"  ({pred} {' '.join(args)})")
    for (fname, fargs), fval in sorted(world.fluents.items()):
        lines.append(f"  (= ({fname} {' '.join(fargs)}) {fval})")
    if goal_expr and static_facts is not None:
        goal_met = check_goal(world, static_facts, goal_expr, enable_numeric)
        lines.append(f"Goal satisfied: {goal_met}")
    return "\n".join(lines)
