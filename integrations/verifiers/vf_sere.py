"""SERE integration for Verifiers/Prime Environments."""

from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path

try:
    import verifiers as vf
    from verifiers.types import Messages, State
    from datasets import Dataset
except ImportError as e:
    raise ImportError(
        "Verifiers integration requires verifiers. "
        "Install with: uv sync --extra verifiers"
    ) from e

from sere.io.task_loader import load_task
from sere.io.pddl_loader import load_agentic_task
from sere.core.pddl_env.run_mode import RunMode
from integrations.verifiers.parser import (
    parse_pddl_actions,
    format_multi_agent_prompt,
)
from integrations.verifiers.dataset import discover_tasks, get_available_domains


__all__ = [
    "load_environment",
    "SereEnv",
    "AgenticSereEnv",
    "parse_pddl_actions",
    "format_multi_agent_prompt",
    "discover_tasks",
    "get_available_domains",
]


# ---------------------------------------------------------------------------
# Rubric helpers
# ---------------------------------------------------------------------------

def _extract_outcome(state: dict) -> str | None:
    for step in reversed(state.get("trajectory", [])):
        extras = step.get("extras") or {}
        info = extras.get("sere_info")
        if isinstance(info, dict) and info.get("outcome") is not None:
            return str(info["outcome"])
    return None


def sum_step_rewards(state: dict, **kwargs) -> float:
    return float(
        sum(
            float(step.get("reward", 0.0) or 0.0)
            for step in state.get("trajectory", [])
        )
    )


def task_success(state: dict, **kwargs) -> float:
    outcome = _extract_outcome(state)
    return 1.0 if outcome and outcome.lower() == "success" else 0.0


def _outcome_is(state: dict, label: str) -> float:
    outcome = _extract_outcome(state)
    if not outcome:
        return 0.0
    return 1.0 if outcome.lower() == label.lower() else 0.0


def outcome_invalid_move(state: dict, **kwargs) -> float:
    return _outcome_is(state, "invalid_move")


def outcome_timeout(state: dict, **kwargs) -> float:
    return _outcome_is(state, "timeout")


def outcome_out_of_energy(state: dict, **kwargs) -> float:
    return _outcome_is(state, "out_of_energy")


def outcome_failed(state: dict, **kwargs) -> float:
    return _outcome_is(state, "failed")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SereEnv(vf.MultiTurnEnv):
    """Native MultiTurnEnv for SERE symbolic reasoning tasks."""

    def __init__(
        self,
        sere_env_kwargs: Dict[str, Any],
        run_mode: RunMode = RunMode.INTERACTIVE,
        system_prompt_override: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sere_env_kwargs = sere_env_kwargs
        self.run_mode = run_mode
        self.system_prompt_override = system_prompt_override

    async def setup_state(self, state: State) -> State:
        """Load SERE task and build the real prompt with system prompt + observation."""
        task_path = state["info"]["task_path"]
        seed = state["info"].get("seed", 0)

        env, meta = load_task(
            domain_path=None,
            task_path=task_path,
            run_mode=self.run_mode,
            seed=seed,
            **self.sere_env_kwargs,
        )
        obs, sere_info = env.reset(seed=seed)

        state["sere_env"] = env
        state["sere_done"] = False

        # Build the real prompt (replacing placeholder)
        sys_prompt = self.system_prompt_override or sere_info.get("system_prompt", "")
        prompt: list[dict[str, str]] = []
        if sys_prompt:
            prompt.append({"role": "system", "content": sys_prompt})
        prompt.append({"role": "user", "content": obs})
        state["prompt"] = prompt

        return state

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Parse action from model output, step the SERE env, return observation."""
        env = state["sere_env"]

        # Extract the model's last message
        raw_text = ""
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    raw_text = msg.get("content", "")
                    break

        # Extract PDDL action(s) — if parsing fails, pass raw text to env
        # (the env's own error handling will produce an "Invalid" message)
        try:
            action = parse_pddl_actions(raw_text)
            if isinstance(action, list):
                action_text = "".join(action)
            else:
                action_text = action
        except ValueError:
            action_text = raw_text

        obs, reward, done, info = env.step(action_text)

        # Record reward + info on the last trajectory step
        if state["trajectory"]:
            state["trajectory"][-1]["reward"] = reward
            state["trajectory"][-1]["extras"]["sere_info"] = info

        state["sere_done"] = done

        response: Messages = [{"role": "user", "content": obs}]

        if done:
            state["final_env_response"] = response

        return response

    @vf.stop
    async def episode_done(self, state: State) -> bool:
        return state.get("sere_done", False)

    @vf.cleanup
    async def cleanup_env(self, state: State) -> None:
        state.pop("sere_env", None)


# ---------------------------------------------------------------------------
# Agentic environment (tool-based plan validation)
# ---------------------------------------------------------------------------

class AgenticSereEnv(vf.MultiTurnEnv):
    """
    miniSWE-style PDDL environment.

    Workspace: domain.pddl (ro), problem.pddl (ro), plan.pddl (rw).
    Tools: read_file, write_file, validate, simulate.
    Only full validate (no up_to_step) can end the episode.
    """

    def __init__(
        self,
        agentic_kwargs: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agentic_kwargs = agentic_kwargs

    async def setup_state(self, state: State) -> State:
        """Load task and build system prompt with domain + problem PDDL."""
        task_path = state["info"]["task_path"]
        domain_dir = state["info"]["domain_dir"]

        agentic_env, meta = load_agentic_task(
            domain_dir,
            task_path,
            **self.agentic_kwargs,
        )

        state["agentic_env"] = agentic_env
        state["sere_done"] = False

        prompt: list[dict[str, str]] = [
            {"role": "system", "content": agentic_env.system_prompt()},
            {"role": "user", "content": (
                "Solve this PDDL planning problem. "
                "Read the domain and problem files, write a plan, and validate it."
            )},
        ]
        state["prompt"] = prompt
        state["tools"] = agentic_env.tool_schemas()

        return state

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Dispatch tool calls to the agentic env."""
        agentic_env = state["agentic_env"]

        # Extract tool call(s) from the model's last message
        tool_call = _extract_tool_call(messages)

        if tool_call is None:
            # Model didn't call a tool — nudge it
            return [{"role": "user", "content": (
                "Use the available tools. Start by reading the domain and "
                "problem files, then write a plan and validate it."
            )}]

        tool_name, tool_args, tool_call_id = tool_call
        feedback, done = agentic_env.handle_tool_call(tool_name, tool_args)

        # Record outcome on trajectory for validate calls
        if state["trajectory"] and tool_name == "validate":
            outcome = "success" if agentic_env.solved else "failed"
            state["trajectory"][-1]["reward"] = 1.0 if agentic_env.solved else 0.0
            state["trajectory"][-1]["extras"]["sere_info"] = {
                "outcome": outcome if done else None,
                "attempts": agentic_env.attempts,
                "solved": agentic_env.solved,
            }

        state["sere_done"] = done

        response: Messages = [
            {"role": "tool", "content": feedback, "tool_call_id": tool_call_id or tool_name},
        ]

        if done:
            state["final_env_response"] = response

        return response

    @vf.stop
    async def episode_done(self, state: State) -> bool:
        return state.get("sere_done", False)

    @vf.cleanup
    async def cleanup_env(self, state: State) -> None:
        env = state.pop("agentic_env", None)
        if env is not None:
            env.cleanup()


def _extract_tool_call(
    messages: Messages,
) -> tuple[str, dict, str | None] | None:
    """
    Extract the first tool call from the model's messages.
    Returns (tool_name, arguments_dict, tool_call_id) or None.
    """
    import json

    if not isinstance(messages, list):
        return None

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue

        # Standard tool_calls format
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            if name:
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tc_id = tc.get("id")
                return name, args, tc_id

        # Fallback: if assistant sent raw text with PDDL actions, treat as validate_plan
        if msg.get("role") == "assistant" and not tool_calls:
            content = msg.get("content", "")
            if content and "(" in content:
                return "validate_plan", {"plan": content}, None

    return None


# ---------------------------------------------------------------------------
# load_environment
# ---------------------------------------------------------------------------

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
    step_penalty: float = 0.0,
    invalid_penalty: float = 0.0,
    illegal_move_retries: int = 100,
    time_limit: float | None = None,
    enable_reward_shaping: bool = False,
    reward_shaping: Dict[str, Any] | None = None,
    # Observation formatting
    show_domain_pddl: bool = True,
    show_affordances: bool = True,
    # Advanced
    system_prompt: str | None = None,
    run_mode: RunMode = RunMode.INTERACTIVE,
    env_kwargs: Dict[str, Any] | None = None,
    **kwargs,
) -> SereEnv:
    """
    Load SERE environment for Verifiers RL training.

    Returns a native MultiTurnEnv — no gym wrapper, no upfront env resets.
    Tasks are loaded lazily at rollout time.

    Args:
        task_paths: Explicit list of task paths — YAML (e.g., "kitchen/t01.yaml")
                    or PDDL (e.g., absolute path to a .pddl problem file).
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
        step_penalty: Reward penalty per action (default: 0, outcome-only).
        invalid_penalty: Penalty for invalid actions (default: 0, outcome-only).
        illegal_move_retries: Number of retries allowed on invalid actions
            before terminating (default: 100, effectively unlimited).
        time_limit: Optional time limit for episodes (in time units).
        enable_reward_shaping: If True, allow task-defined reward shaping milestones.
        reward_shaping: Dict with shaping config.

        show_domain_pddl: Include raw PDDL domain in system prompt (default: True).
        show_affordances: Show available actions in observations (default: True).

        system_prompt: Custom system prompt. If None, uses per-task SERE system prompt.
        run_mode: SERE run mode (default: INTERACTIVE).
        env_kwargs: Additional arguments passed to SERE's load_task().
        **kwargs: Additional arguments passed to SereEnv / MultiTurnEnv.

    Returns:
        SereEnv instance ready for training/evaluation

    Examples:
        >>> env = load_environment(domains=["kitchen"])
        >>> env = load_environment(domains=["kitchen"], show_domain_pddl=True)
    """
    # Discover tasks
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

    # Build dataset: just metadata, no env resets
    train_rows = []
    eval_rows = []
    if eval_episodes_per_task is None:
        eval_episodes_per_task = episodes_per_task
    num_tasks = len(task_paths)

    for ep in range(episodes_per_task):
        for i, task_path in enumerate(task_paths):
            domain = _infer_domain(task_path)
            row = {
                "prompt": [{"role": "user", "content": "Loading task..."}],
                "answer": task_path,
                "task": domain,
                "info": {
                    "task_path": task_path,
                    "domain": domain,
                    "seed": seed + ep * num_tasks + i,
                },
            }
            train_rows.append(row)

    for ep in range(eval_episodes_per_task):
        for i, task_path in enumerate(task_paths):
            domain = _infer_domain(task_path)
            row = {
                "prompt": [{"role": "user", "content": "Loading task..."}],
                "answer": task_path,
                "task": domain,
                "info": {
                    "task_path": task_path,
                    "domain": domain,
                    "seed": seed + (episodes_per_task + ep) * num_tasks + i,
                },
            }
            eval_rows.append(row)

    dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    # Build SERE environment configuration
    sere_env_kwargs = dict(env_kwargs or {})
    sere_env_kwargs.setdefault("enable_numeric", enable_numeric)
    sere_env_kwargs.setdefault("enable_conditional", enable_conditional)
    sere_env_kwargs.setdefault("enable_durations", enable_durations)
    sere_env_kwargs.setdefault("enable_stochastic", enable_stochastic)
    sere_env_kwargs.setdefault("step_penalty", step_penalty)
    sere_env_kwargs.setdefault("invalid_penalty", invalid_penalty)
    sere_env_kwargs.setdefault("illegal_move_retries", illegal_move_retries)

    if time_limit is not None:
        sere_env_kwargs.setdefault("time_limit", time_limit)

    if reward_shaping is not None:
        sere_env_kwargs["reward_shaping"] = reward_shaping
    elif "reward_shaping" not in sere_env_kwargs and not enable_reward_shaping:
        sere_env_kwargs["reward_shaping"] = None

    if max_episode_steps is not None:
        sere_env_kwargs.setdefault("max_steps", max_episode_steps)

    # Formatter config
    formatter_config = sere_env_kwargs.get("formatter_config", {})
    formatter_config.setdefault("show_domain_pddl", show_domain_pddl)
    formatter_config.setdefault("show_affordances", show_affordances)
    formatter_config.setdefault("show_goal", True)
    formatter_config.setdefault("show_fluents", True)
    formatter_config.setdefault("show_messages", True)
    sere_env_kwargs["formatter_config"] = formatter_config

    # Rubric
    rubric = kwargs.pop("rubric", None)
    if rubric is None:
        rubric = vf.Rubric(funcs=[task_success], weights=[1.0])
    if hasattr(rubric, "add_metric"):
        rubric.add_metric(sum_step_rewards)
        rubric.add_metric(outcome_invalid_move)
        rubric.add_metric(outcome_timeout)
        rubric.add_metric(outcome_out_of_energy)
        rubric.add_metric(outcome_failed)

    return SereEnv(
        sere_env_kwargs=sere_env_kwargs,
        run_mode=run_mode,
        system_prompt_override=system_prompt,
        dataset=dataset,
        eval_dataset=eval_dataset,
        max_turns=max_episode_steps or 30,
        rubric=rubric,
        message_type="chat",
        **kwargs,
    )


def _infer_domain(task_path: str) -> str:
    """Infer domain name from a task path."""
    p = Path(task_path)
    if task_path.endswith(".pddl"):
        # e.g., .../pddl/blocksworld/problems/instance-1.pddl → blocksworld
        for part in p.parts:
            if part == "problems":
                idx = p.parts.index(part)
                if idx > 0:
                    return p.parts[idx - 1]
        return p.parent.name
    # YAML: kitchen/t01.yaml → kitchen
    return p.parts[0] if p.parts else "unknown"


def _infer_domain_dir(task_path: str) -> str:
    """Infer domain directory from a PDDL problem path."""
    p = Path(task_path)
    # .../pddl/blocksworld/problems/instance-1.pddl → .../pddl/blocksworld
    if p.parent.name == "problems":
        return str(p.parent.parent)
    return str(p.parent)


# ---------------------------------------------------------------------------
# load_agentic_environment
# ---------------------------------------------------------------------------

def load_agentic_environment(
    # Task selection
    domains: List[str] | None = None,
    num_tasks_per_domain: int | None = None,
    # Episode configuration
    episodes_per_task: int = 1,
    eval_episodes_per_task: int | None = None,
    seed: int = 0,
    # Agentic options
    max_attempts: int = 8,
    enable_numeric: bool = False,
    enable_conditional: bool = False,
    # Advanced
    **kwargs,
) -> AgenticSereEnv:
    """
    Load an agentic PDDL planning environment.

    The LLM receives domain.pddl + problem.pddl and uses a validate_plan
    tool to submit complete plans. Returns an AgenticSereEnv.
    """
    # Discover PDDL tasks only
    task_paths = discover_tasks(
        domains=domains,
        num_tasks_per_domain=num_tasks_per_domain,
        include_multi_agent=False,
        include_pddl=True,
        include_yaml=False,
    )

    if not task_paths:
        raise ValueError(f"No PDDL tasks found for domains: {domains}")

    # Build dataset
    train_rows = []
    eval_rows = []
    if eval_episodes_per_task is None:
        eval_episodes_per_task = episodes_per_task
    num_tasks = len(task_paths)

    for ep in range(episodes_per_task):
        for i, task_path in enumerate(task_paths):
            domain = _infer_domain(task_path)
            domain_dir = _infer_domain_dir(task_path)
            row = {
                "prompt": [{"role": "user", "content": "Loading task..."}],
                "answer": task_path,
                "task": domain,
                "info": {
                    "task_path": task_path,
                    "domain_dir": domain_dir,
                    "domain": domain,
                    "seed": seed + ep * num_tasks + i,
                },
            }
            train_rows.append(row)

    for ep in range(eval_episodes_per_task):
        for i, task_path in enumerate(task_paths):
            domain = _infer_domain(task_path)
            domain_dir = _infer_domain_dir(task_path)
            row = {
                "prompt": [{"role": "user", "content": "Loading task..."}],
                "answer": task_path,
                "task": domain,
                "info": {
                    "task_path": task_path,
                    "domain_dir": domain_dir,
                    "domain": domain,
                    "seed": seed + (episodes_per_task + ep) * num_tasks + i,
                },
            }
            eval_rows.append(row)

    dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    # Agentic env kwargs
    agentic_kwargs = {
        "max_attempts": max_attempts,
        "enable_numeric": enable_numeric,
        "enable_conditional": enable_conditional,
    }

    # Rubric: binary task success
    rubric = kwargs.pop("rubric", None)
    if rubric is None:
        rubric = vf.Rubric(funcs=[task_success], weights=[1.0])

    return AgenticSereEnv(
        agentic_kwargs=agentic_kwargs,
        dataset=dataset,
        eval_dataset=eval_dataset,
        max_turns=max_attempts,
        rubric=rubric,
        message_type="chat",
        **kwargs,
    )
