"""SERE integration for Verifiers/Prime Environments."""

from __future__ import annotations

from typing import List, Dict, Any
from datasets import Dataset

try:
    import verifiers as vf
    from verifiers.envs.experimental.gym_env import GymEnv, normalize_reset
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


class SereGymEnv(GymEnv):
    """GymEnv with per-episode system prompts sourced from SERE task info."""

    def __init__(self, *args, system_prompt: str | None = None, **kwargs):
        self._system_prompt_override = system_prompt
        super().__init__(*args, system_prompt=system_prompt, **kwargs)

    def gym_to_hf(self) -> tuple[Dataset, Dataset | None]:
        train_rows = []
        eval_rows = []
        total = self.num_train_episodes + self.num_eval_episodes
        env = self.env_cls(**self.env_kwargs)

        try:
            for i in range(total):
                obs, info = normalize_reset(env.reset(seed=self.seed + i))
                question = self.obs_to_text(obs)
                sys_prompt = self._system_prompt_override or info.get("system_prompt")

                if self.message_type == "completion":
                    row = {"prompt": question, "answer": str(self.seed + i)}
                else:
                    prompt = []
                    if sys_prompt:
                        prompt.append({"role": "system", "content": sys_prompt})
                    prompt.append({"role": "user", "content": question})
                    row = {"prompt": prompt, "answer": str(self.seed + i)}

                if i < self.num_train_episodes:
                    train_rows.append(row)
                else:
                    eval_rows.append(row)
        finally:
            close_fn = getattr(env, "close", None)
            if close_fn is not None:
                close_fn()

        dataset = Dataset.from_list(train_rows)
        eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
        return dataset, eval_dataset


def load_environment(
    # Task selection
    task_paths: List[str] | None = None,
    domains: List[str] | None = None,
    num_tasks_per_domain: int | None = None,
    include_multi_agent: bool = True,
    # Episode configuration
    episodes_per_task: int = 1,
    eval_episodes_per_task: int | None = None,
    max_episode_steps: int | None = None,
    seed: int = 0,
    # SERE environment options
    enable_numeric: bool = True,
    enable_conditional: bool = True,
    enable_durations: bool = True,
    enable_stochastic: bool = False,
    step_penalty: float = -0.01,
    invalid_penalty: float = -0.1,
    time_limit: float | None = None,
    reward_shaping: Dict[str, Any] | None = None,
    # Observation formatting
    display_nl: bool = True,
    show_affordances: bool = True,
    # Advanced
    system_prompt: str | None = None,
    run_mode: RunMode = RunMode.INTERACTIVE,
    env_kwargs: Dict[str, Any] | None = None,
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
        seed: Random seed for reproducibility.

        enable_numeric: Enable numeric fluents (battery level, temperature, etc.).
        enable_conditional: Enable conditional effects in actions.
        enable_durations: Enable time-based action durations.
        enable_stochastic: Enable stochastic action outcomes.
        step_penalty: Reward penalty per action (default: -0.01).
        invalid_penalty: Penalty for invalid actions (default: -0.1).
        time_limit: Optional time limit for episodes (in time units).
        reward_shaping: Dict with shaping config: {"mode": "milestone"|"potential",
                        "milestones": [...], "gamma": 1.0}.

        display_nl: Show natural language descriptions alongside PDDL (default: True).
        show_affordances: Show available actions in observations (default: True).

        system_prompt: Custom system prompt. If None, uses per-task SERE system prompt.
        run_mode: SERE run mode (default: INTERACTIVE).
        env_kwargs: Additional arguments passed to SERE's load_task().
                    Advanced options not covered by other parameters.
        **kwargs: Additional arguments passed to GymEnv.

    Returns:
        GymEnv instance ready for training/evaluation

    Examples:
        >>> # Basic usage - all kitchen tasks
        >>> env = load_environment(domains=["kitchen"])

        >>> # Disable natural language, only show PDDL
        >>> env = load_environment(
        ...     domains=["kitchen"],
        ...     display_nl=False,
        ...     show_affordances=False
        ... )

        >>> # Enable stochastic actions with custom penalties
        >>> env = load_environment(
        ...     domains=["kitchen"],
        ...     enable_stochastic=True,
        ...     step_penalty=-0.05,
        ...     invalid_penalty=-0.5
        ... )

        >>> # Custom reward shaping
        >>> env = load_environment(
        ...     domains=["kitchen"],
        ...     reward_shaping={
        ...         "mode": "milestone",
        ...         "milestones": [
        ...             {"expr": "(has-hot-water ?c)", "reward": 0.5, "once": True}
        ...         ]
        ...     }
        ... )
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

    # Build SERE environment configuration
    sere_env_kwargs = env_kwargs or {}

    # Add environment behavior options
    sere_env_kwargs.setdefault("enable_numeric", enable_numeric)
    sere_env_kwargs.setdefault("enable_conditional", enable_conditional)
    sere_env_kwargs.setdefault("enable_durations", enable_durations)
    sere_env_kwargs.setdefault("enable_stochastic", enable_stochastic)
    sere_env_kwargs.setdefault("step_penalty", step_penalty)
    sere_env_kwargs.setdefault("invalid_penalty", invalid_penalty)

    if time_limit is not None:
        sere_env_kwargs.setdefault("time_limit", time_limit)

    if reward_shaping is not None:
        sere_env_kwargs.setdefault("reward_shaping", reward_shaping)

    # Build observation formatter configuration
    formatter_config = sere_env_kwargs.get("formatter_config", {})
    formatter_config.setdefault("display_nl", display_nl)
    formatter_config.setdefault("show_affordances", show_affordances)
    # Always show goal, fluents, and messages - these should not be configurable
    formatter_config.setdefault("show_goal", True)
    formatter_config.setdefault("show_fluents", True)
    formatter_config.setdefault("show_messages", True)
    sere_env_kwargs["formatter_config"] = formatter_config

    # Create GymEnv with SERE wrapper
    env = SereGymEnv(
        # SERE-specific
        env_cls=SereGymWrapper,
        env_kwargs={
            "task_paths": task_paths,
            "env_kwargs": sere_env_kwargs,
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
