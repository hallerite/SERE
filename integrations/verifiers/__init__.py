"""SERE integration for Verifiers/Prime Environments."""

from __future__ import annotations

from typing import List, Dict, Any

try:
    import verifiers as vf
    from verifiers.envs.experimental.gym_env import GymEnv
except ImportError as e:
    raise ImportError(
        "Verifiers integration requires verifiers. "
        "Install with: uv sync --extra verifiers"
    ) from e

from sere.core.pddl_env.run_mode import RunMode
from integrations.verifiers.wrapper import SereGymWrapper
from integrations.verifiers.parser import (
    parse_pddl_actions,
    format_multi_agent_prompt,
)
from integrations.verifiers.dataset import discover_tasks, get_available_domains


__all__ = [
    "load_environment",
    "SereGymWrapper",
    "parse_pddl_actions",
    "format_multi_agent_prompt",
    "discover_tasks",
    "get_available_domains",
]


def load_environment(
    task_paths: List[str] | None = None,
    domains: List[str] | None = None,
    num_tasks_per_domain: int | None = None,
    include_multi_agent: bool = True,
    episodes_per_task: int = 1,
    eval_episodes_per_task: int | None = None,
    max_episode_steps: int | None = None,
    system_prompt: str | None = None,
    run_mode: RunMode = RunMode.INTERACTIVE,
    env_kwargs: Dict[str, Any] | None = None,
    seed: int = 0,
    **kwargs,
) -> GymEnv:
    """
    Load SERE environment for Verifiers RL training.

    This function creates a GymEnv that wraps SERE's symbolic reasoning tasks,
    making them compatible with the Verifiers training infrastructure.

    Args:
        task_paths: Explicit list of task YAML paths (e.g., ["kitchen/t01.yaml"]).
                    If provided, other discovery arguments are ignored.
        domains: List of domains to include (e.g., ["kitchen", "assembly"]).
                 If None, discovers all available domains.
        num_tasks_per_domain: Limit number of tasks per domain.
        include_multi_agent: Whether to include multi-agent tasks (default: True).
        episodes_per_task: Number of training episodes per task (default: 1).
        eval_episodes_per_task: Number of evaluation episodes per task.
                                Defaults to same as episodes_per_task.
        max_episode_steps: Maximum steps per episode (overrides task max_steps).
        system_prompt: Custom system prompt. If None, uses default.
        run_mode: SERE run mode (default: INTERACTIVE).
        env_kwargs: Additional arguments passed to SERE's load_task().
        seed: Random seed for reproducibility.
        **kwargs: Additional arguments passed to GymEnv.

    Returns:
        GymEnv instance ready for training/evaluation

    Example:
        >>> # Load all kitchen tasks
        >>> env = load_environment(domains=["kitchen"])
        >>>
        >>> # Load specific tasks
        >>> env = load_environment(
        ...     task_paths=["kitchen/t01_one_step_steep.yaml",
        ...                 "assembly/t01_one_step_fasten.yaml"]
        ... )
        >>>
        >>> # Load all tasks from all domains (default)
        >>> env = load_environment()
    """
    # Discover tasks if not explicitly provided
    if task_paths is None:
        task_paths = discover_tasks(
            domains=domains,
            num_tasks_per_domain=num_tasks_per_domain,
            include_multi_agent=include_multi_agent,
        )

    if not task_paths:
        raise ValueError(
            f"No tasks found. Domains: {domains}, "
            f"include_multi_agent: {include_multi_agent}"
        )

    # Calculate episode counts
    num_tasks = len(task_paths)
    num_train_episodes = num_tasks * episodes_per_task

    if eval_episodes_per_task is None:
        eval_episodes_per_task = episodes_per_task
    num_eval_episodes = num_tasks * eval_episodes_per_task

    # Build default system prompt
    if system_prompt is None:
        system_prompt = _build_default_system_prompt()

    # Create GymEnv with SERE wrapper
    env = GymEnv(
        # SERE-specific
        env_cls=SereGymWrapper,
        env_kwargs={
            "task_paths": task_paths,
            "env_kwargs": env_kwargs or {},
            "run_mode": run_mode,
        },
        action_parser=parse_pddl_actions,
        obs_to_text=lambda obs: obs,  # SERE obs is already a string
        # Episode configuration
        num_train_episodes=num_train_episodes,
        num_eval_episodes=num_eval_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
        # Verifiers configuration
        system_prompt=system_prompt,
        message_type="chat",
        **kwargs,
    )

    return env


def _build_default_system_prompt() -> str:
    """Build default system prompt for SERE tasks."""
    return """You are solving symbolic reasoning tasks in a PDDL-style environment.

Each task presents you with:
- Current state of the world
- Goal to achieve
- Available actions

Respond with valid PDDL actions in S-expression format.

Examples:
- Single action: (move r1 kitchen pantry)
- Multiple actions (multi-agent): (move r1 kitchen pantry)
                                    (pick-up r2 leaf)

For multi-agent tasks, provide one action per robot. Use (idle <robot>) if a robot should not act.
"""
