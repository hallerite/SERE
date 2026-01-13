"""PDDL action parsing for Verifiers integration."""

from __future__ import annotations

import re
from typing import List

from sere.pddl.sexpr import parse_one, SExprError


def parse_pddl_actions(text: str) -> str | List[str]:
    """
    Extract PDDL S-expressions from model output.

    Supports:
    - Raw S-expression: "(move r1 kitchen pantry)"
    - With explanation: "I'll move to pantry. (move r1 kitchen pantry)"
    - Multiple actions (multi-agent): "(move r1 kitchen)(pick-up r2 leaf)"
    - Embedded in text with multiple attempts

    Args:
        text: Model output text

    Returns:
        For single action: action string
        For multiple actions: list of action strings

    Raises:
        ValueError: If no valid PDDL S-expression found
    """
    # Find all S-expressions (balanced parentheses)
    actions = []

    # Pattern to find S-expressions
    # Matches (action-name arg1 arg2 ...)
    pattern = r'\([a-zA-Z][a-zA-Z0-9_\-]*(?:\s+[a-zA-Z0-9_\-]+)*\)'

    matches = re.findall(pattern, text)

    for match in matches:
        try:
            # Validate it's a proper S-expression
            parse_one(match)
            actions.append(match)
        except SExprError:
            # Skip invalid S-expressions
            continue

    if not actions:
        raise ValueError(f"No valid PDDL S-expression found in: {text[:200]}")

    # Return single action as string, multiple as list
    if len(actions) == 1:
        return actions[0]
    return actions


def format_multi_agent_prompt(robot_ids: List[str]) -> str:
    """
    Generate system prompt instructions for multi-agent tasks.

    Args:
        robot_ids: List of robot identifiers (e.g., ["r1", "r2", "r3"])

    Returns:
        Additional system prompt text explaining multi-agent format
    """
    robots_str = ", ".join(robot_ids)
    example_actions = "\n".join([
        f"(move {robot_ids[0]} kitchen pantry)"
        if i == 0
        else f"(idle {rid})"
        for i, rid in enumerate(robot_ids[:3])  # Show first 3 robots
    ])

    return f"""
This is a multi-agent task with robots: {robots_str}.

IMPORTANT: You must provide one action per robot in each turn.
Format your response with multiple PDDL actions, one per robot:

Example:
{example_actions}

Use (idle <robot>) if a robot should not act this turn.
"""


def validate_action(action_text: str) -> bool:
    """
    Validate that text contains a valid PDDL S-expression.

    Args:
        action_text: Text to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        parse_pddl_actions(action_text)
        return True
    except ValueError:
        return False
