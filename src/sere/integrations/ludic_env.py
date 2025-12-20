from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ludic.envs.env import LudicEnv
from ludic.types import Info, StepOutcome

from sere.core.pddl_env.env import PDDLEnv
from sere.core.pddl_env.run_mode import RunMode
from sere.core.pddl_env import planning
from sere.io.task_loader import load_task
from sere.integrations.ludic_parser import format_action


class SereLudicEnv(LudicEnv[str, str, str]):
    """
    Ludic-compatible wrapper for SERE's PDDLEnv.

    Multi-agent behavior:
      - One agent per robot (robot symbols sorted for stability).
      - Each agent must provide exactly one action per step.
    """

    def __init__(
        self,
        env: PDDLEnv,
        *,
        agent_ids: Optional[List[str]] = None,
        force_interactive: bool = True,
        system_prompt_suffix: Optional[str] = None,
    ) -> None:
        self._env = env
        if force_interactive and env.run_mode != RunMode.INTERACTIVE:
            env.run_mode = RunMode.INTERACTIVE

        self._agent_ids = agent_ids or self._default_agent_ids()
        if not self._env.multi_agent and agent_ids and len(agent_ids) != 1:
            raise ValueError("Single-agent SERE envs require exactly one agent_id.")
        if self._env.multi_agent and not self._agent_ids:
            raise ValueError("Multi-agent SERE env requires at least one robot.")

        self._last_obs = ""
        self._suggested_sysprompt: Optional[str] = None
        self._system_prompt_suffix = (
            system_prompt_suffix.strip() if system_prompt_suffix else None
        )

    @property
    def env(self) -> PDDLEnv:
        return self._env

    @property
    def agent_ids(self) -> List[str]:
        return list(self._agent_ids)

    @property
    def active_agents(self) -> List[str]:
        if self._env.done:
            return []
        return list(self._agent_ids)

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return self._suggested_sysprompt

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, Tuple[str, Info]]:
        obs, info = self._env.reset(seed=seed)
        self._last_obs = obs
        self._suggested_sysprompt = self._apply_prompt_suffix(info.get("system_prompt"))
        return {agent_id: (obs, dict(info)) for agent_id in self._agent_ids}

    def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
        if self._env.done:
            raise RuntimeError("Episode finished. Call reset().")

        if self._env.multi_agent:
            return self._step_multi_agent(actions)
        return self._step_single_agent(actions)

    def current_obs(self) -> Dict[str, str]:
        return {agent_id: self._last_obs for agent_id in self._agent_ids}

    def _step_single_agent(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
        agent_id = self._agent_ids[0] if self._agent_ids else "agent_0"
        action = actions.get(agent_id, "")
        obs, reward, done, info = self._env.step(action)
        return self._build_outcomes(obs, reward, done, info)

    def _step_multi_agent(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
        missing = [agent_id for agent_id in self.active_agents if agent_id not in actions]
        if missing:
            obs, reward, done, info = self._env._illegal(
                f"Missing actions for agents: {missing}.",
                {"missing_actions": list(missing)},
            )
            return self._build_outcomes(obs, reward, done, info)

        plan = []
        for agent_id in self._agent_ids:
            raw_action = actions.get(agent_id, "")
            try:
                parsed = planning.parse_actions(raw_action)
            except Exception as exc:
                obs, reward, done, info = self._env._illegal(
                    f"Invalid action for {agent_id}: {exc}",
                    {"agent_id": agent_id},
                )
                return self._build_outcomes(obs, reward, done, info)

            if len(parsed) != 1:
                obs, reward, done, info = self._env._illegal(
                    f"Agent {agent_id} must provide exactly one action.",
                    {"agent_id": agent_id},
                )
                return self._build_outcomes(obs, reward, done, info)

            name, args = parsed[0]
            plan.append(format_action(name, args))

        action_block = "".join(plan)
        obs, reward, done, info = self._env.step(action_block)
        return self._build_outcomes(obs, reward, done, info)

    def _build_outcomes(
        self,
        obs: str,
        reward: float,
        done: bool,
        info: Info,
    ) -> Dict[str, StepOutcome]:
        terminated, truncated = self._classify_done(done, info)
        self._last_obs = obs
        return {
            agent_id: StepOutcome(
                obs=obs,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=dict(info),
            )
            for agent_id in self._agent_ids
        }

    def _classify_done(self, done: bool, info: Info) -> Tuple[bool, bool]:
        if not done:
            return False, False
        outcome = str(info.get("outcome", "")).lower()
        reason = str(info.get("reason", "")).lower()
        if outcome == "timeout" or reason in {"max_steps_exceeded", "time_limit_exceeded"}:
            return False, True
        return True, False

    def _default_agent_ids(self) -> List[str]:
        if self._env.multi_agent:
            return self._sorted_robot_ids()
        return ["agent_0"]

    def _sorted_robot_ids(self) -> List[str]:
        robots: List[str] = []
        for sym, types in self._env.world.objects.items():
            if isinstance(types, str):
                type_list = [types]
            else:
                type_list = list(types or [])
            if any(t.lower() == "robot" for t in type_list):
                robots.append(sym)
        return sorted(robots)

    def _apply_prompt_suffix(self, base_prompt: Optional[str]) -> Optional[str]:
        suffix = self._system_prompt_suffix
        if not suffix:
            return base_prompt
        if base_prompt:
            return f"{base_prompt}\n\n{suffix}"
        return suffix


def make_ludic_env(
    domain_path: Optional[str],
    task_path: str,
    *,
    plugins=None,
    agent_ids: Optional[List[str]] = None,
    force_interactive: bool = True,
    system_prompt_suffix: Optional[str] = None,
    **env_kwargs,
) -> Tuple[SereLudicEnv, dict]:
    env, meta = load_task(domain_path, task_path, plugins=plugins, **env_kwargs)
    return SereLudicEnv(
        env,
        agent_ids=agent_ids,
        force_interactive=force_interactive,
        system_prompt_suffix=system_prompt_suffix,
    ), meta
