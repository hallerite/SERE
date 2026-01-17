"""Gym-compatible wrapper for SERE PDDLEnv."""

from __future__ import annotations

from typing import Dict, Any, Tuple, List

from sere.io.task_loader import load_task
from sere.core.pddl_env import PDDLEnv
from sere.core.pddl_env.run_mode import RunMode


class SereGymWrapper:
    """
    Gym-compatible wrapper for SERE tasks.

    Makes SERE task collection look like a single Gym environment where
    each reset(seed=i) loads a different task.

    This wrapper implements the StepResetEnv protocol required by GymEnv:
    - reset(seed: int) -> (obs, info)
    - step(action) -> (obs, reward, done, truncated, info)
    """

    def __init__(
        self,
        task_paths: List[str],
        env_kwargs: Dict[str, Any] | None = None,
        run_mode: RunMode = RunMode.INTERACTIVE,
    ):
        """
        Initialize SERE Gym wrapper.

        Args:
            task_paths: List of SERE task YAML paths
            env_kwargs: Additional arguments to pass to load_task()
            run_mode: SERE run mode (default: INTERACTIVE)
        """
        self.task_paths = task_paths
        self.env_kwargs = env_kwargs or {}
        self.run_mode = run_mode
        self.current_env: PDDLEnv | None = None
        self.current_task_info: Dict[str, Any] = {}

    def reset(self, seed: int = 0, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Reset environment by loading a task.

        Uses seed as index into task_paths (with wraparound).

        Args:
            seed: Seed/index for task selection
            **kwargs: Additional reset arguments

        Returns:
            (observation, info) tuple where:
            - observation: Initial observation string
            - info: Dict with task metadata + SERE info
        """
        # Use seed as index (with wraparound for multiple episodes per task)
        task_idx = seed % len(self.task_paths)
        task_path = self.task_paths[task_idx]

        # Load SERE task
        self.current_env, meta = load_task(
            domain_path=None,
            task_path=task_path,
            run_mode=self.run_mode,
            seed=seed,  # Pass seed to SERE for reproducibility
            **self.env_kwargs,
        )

        # Get initial observation
        obs, sere_info = self.current_env.reset(seed=seed)

        # Build combined info dict
        self.current_task_info = {
            "task_path": task_path,
            "task_id": meta.get("id", ""),
            "task_name": meta.get("name", ""),
            "domain": meta.get("domain", ""),
            "multi_agent": self.current_env.multi_agent,  # Get from env, not meta
            "max_steps": meta.get("max_steps", 40),
            "robot_ids": self._get_robot_ids(),
        }

        info = {**sere_info, **self.current_task_info}

        return obs, info

    def step(self, action: str | List[str]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute action in current SERE environment.

        Args:
            action: PDDL action string (or multiple actions for multi-agent)

        Returns:
            (obs, reward, done, info) tuple (4-tuple, old Gym API)
            GymEnv will normalize this to 5-tuple (obs, reward, done, truncated, info)
        """
        if self.current_env is None:
            raise RuntimeError("Must call reset() before step()")

        if isinstance(action, list):
            if not action:
                action_text = ""
            elif len(action) == 1:
                action_text = str(action[0])
            else:
                action_text = "".join(str(a) for a in action)
        else:
            action_text = str(action)

        # SERE's step returns (obs, reward, done, info)
        obs, reward, done, info = self.current_env.step(action_text)

        # Add task metadata to info
        info = {**info, **self.current_task_info}

        return obs, reward, done, info

    def close(self):
        """Clean up resources (if needed)."""
        self.current_env = None
        self.current_task_info = {}

    def _get_robot_ids(self) -> List[str]:
        """Extract robot IDs from current environment."""
        if self.current_env is None:
            return []

        robots = []
        for sym, types in self.current_env.world.objects.items():
            if isinstance(types, str):
                type_list = [types]
            else:
                type_list = list(types or [])

            if any(t.lower() == "robot" for t in type_list):
                robots.append(sym)

        return sorted(robots)
