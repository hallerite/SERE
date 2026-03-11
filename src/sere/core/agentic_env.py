"""
Agentic PDDL environment: LLM writes complete plans, validator checks them.

The LLM gets domain.pddl + problem.pddl and uses a `validate_plan` tool
to submit plans. The tool returns structured feedback on success or failure.
Guards ensure the goal and initial state cannot be changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from sere.pddl.domain_spec import DomainSpec, Predicate
from sere.pddl.pddl_parser import PDDLProblem
from sere.core.world_state import WorldState
from sere.core.validator import (
    PlanResult,
    check_goal,
    format_plan_feedback,
    validate_plan,
)
from sere.core.pddl_env.planning import parse_actions


@dataclass
class AgenticPDDLEnv:
    """
    Agentic PDDL planning environment.

    The LLM receives domain.pddl + problem.pddl in the system prompt,
    then calls a `validate_plan` tool with a plan. The validator simulates
    the plan from the initial state and returns success or first failure.

    State resets from scratch for each plan attempt — no carryover.
    """

    domain: DomainSpec
    init_world: WorldState
    static_facts: Set[Predicate]
    goal_expr: str
    domain_pddl: str
    problem_pddl: str
    problem_name: str = ""

    # Feature flags (match the problem)
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
            "You have access to a `validate_plan` tool. Call it with your plan "
            "as a sequence of grounded actions:\n"
            "  (action-name arg1 arg2)\n"
            "  (action-name arg1 arg2)\n"
            "  ...\n\n"
            "If validation fails, you will see which step failed and why. "
            "Revise your plan and call the tool again.\n\n"
            f"=== DOMAIN ===\n{self.domain_pddl}\n\n"
            f"=== PROBLEM ===\n{self.problem_pddl}"
        )

    def tool_schema(self) -> Dict[str, Any]:
        """Return the tool definition for `validate_plan`."""
        return {
            "type": "function",
            "function": {
                "name": "validate_plan",
                "description": (
                    "Validate a PDDL plan against the domain and problem. "
                    "Submit a complete sequence of grounded actions. "
                    "Returns success or diagnostic feedback on the first failing step."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "string",
                            "description": (
                                "The plan as a sequence of grounded PDDL actions, "
                                "one per line. Example:\n"
                                "(pick-up A)\n(stack A B)\n(pick-up C)\n(stack C A)"
                            ),
                        },
                    },
                    "required": ["plan"],
                },
            },
        }

    def validate(self, plan_text: str) -> Tuple[str, bool]:
        """
        Parse and validate a plan. Returns (feedback_string, is_done).

        Guards:
        - Plan is parsed fresh each time
        - State always resets from init (no accumulated state)
        - Goal and init are immutable (defined at construction)
        """
        self.attempts += 1

        # Parse the plan
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

        # Validate against a fresh copy of init state
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
