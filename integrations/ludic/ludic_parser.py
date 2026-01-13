from __future__ import annotations

from typing import Tuple

from ludic.parsers import ParseResult, Parser, xml_tag_parser, compose_parsers

from sere.core.pddl_env import planning


def format_action(name: str, args: Tuple[str, ...]) -> str:
    if args:
        return f"({name} {' '.join(args)})"
    return f"({name})"


def pddl_action_parser(
    *,
    success_reward: float = 0.0,
    error_reward: float = -1.0,
) -> Parser:
    """
    Parse exactly one raw PDDL action S-expression.

    Example input: "(move r1 kitchen pantry)"
    """
    def _parser(raw: str) -> ParseResult:
        try:
            plan = planning.parse_actions(raw)
        except Exception as exc:
            return ParseResult(
                action=None,
                reward=error_reward,
                obs=f"Invalid action format: {exc}",
            )

        if len(plan) != 1:
            return ParseResult(
                action=None,
                reward=error_reward,
                obs="Expected exactly one action S-expression.",
            )

        name, args = plan[0]
        return ParseResult(
            action=format_action(name, args),
            reward=success_reward,
            obs=None,
        )

    return _parser


def pddl_action_tag_parser(
    *,
    tag: str = "action",
    success_reward: float = 0.0,
    error_reward: float = -1.0,
) -> Parser:
    """
    Parse a PDDL action wrapped in an XML-like tag.

    Example input: "<action>(move r1 kitchen pantry)</action>"
    """
    tag_parser = xml_tag_parser(
        tag,
        exact=False,
        success_reward=0.0,
        error_reward=error_reward,
    )

    pddl_parser = pddl_action_parser(
        success_reward=success_reward,
        error_reward=error_reward,
    )
    
    return compose_parsers(tag_parser, pddl_parser)
