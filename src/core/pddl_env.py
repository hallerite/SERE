from typing import Any, Dict, List, Optional, Tuple
from ..pddl.domain_spec import DomainSpec
from ..pddl.grounding import parse_grounded, instantiate
from .world_state import WorldState
from .invariants import InvariantPlugin

def extract_tag(s: str, tag: str) -> Optional[str]:
    start, end = f"<{tag}>", f"</{tag}>"
    i, j = s.find(start), s.find(end)
    return None if i < 0 or j < 0 else s[i+len(start):j].strip()

class PDDLEnv:
    def __init__(self, domain: DomainSpec, world: WorldState,
                 static_facts: set, goal: List[tuple],
                 plugins: Optional[List[InvariantPlugin]] = None,
                 max_steps: int = 40, step_penalty: float = -0.01,
                 invalid_penalty: float = -0.1, goal_reward: float = 1.0):
        self.domain, self.world = domain, world
        self.static_facts, self.goal = static_facts, goal
        self.plugins = plugins or []
        self.max_steps = max_steps
        self.step_penalty, self.invalid_penalty, self.goal_reward = step_penalty, invalid_penalty, goal_reward
        self.steps = 0
        self.done = False

    def reset(self):
        self.steps, self.done = 0, False
        errs = self.world.validate_invariants()
        for pl in self.plugins: errs += pl.validate(self.world)
        if errs:
            raise ValueError(f"Invariant errors: {errs}")
        obs = {"observation": self._obs(), "system_prompt": "Provide ONE grounded PDDL action in <move> tags."}
        info = {"problem_pddl": self.world.to_problem_pddl("instance", self.static_facts, self.goal)}
        return obs, info

    def step(self, text: str):
        if self.done: raise RuntimeError("Episode finished. Call reset().")
        info: Dict[str, Any] = {}
        move = extract_tag(text, "move")
        if not move:
            return self._illegal("Missing <move>(...)</move>.", info)

        try:
            name, args = parse_grounded(move)
        except Exception as e:
            return self._illegal(f"Parse error: {e}", info)

        if name not in self.domain.actions:
            return self._illegal(f"Unknown action '{name}'", info)

        act = self.domain.actions[name]
        pre, add, dele = instantiate(self.domain, act, args)

        missing = self.world.check_preconds(pre, extra=self.static_facts)
        if missing:
            return self._illegal(f"Preconditions not satisfied: {missing}", info)

        self.world.apply(add, dele)
        self.steps += 1
        for pl in self.plugins:
            errs = pl.validate(self.world)
            if errs:
                return self._illegal(f"Postcondition violated: {errs}", info)

        reward = self.step_penalty
        if all(self.world.holds(g) for g in self.goal):
            self.done = True
            reward += self.goal_reward
            info["outcome"] = "win"
        elif self.steps >= self.max_steps:
            self.done = True
            info["outcome"] = "loss"
        else:
            info["outcome"] = "ongoing"

        return {"observation": self._obs()}, reward, self.done, info

    def _illegal(self, msg: str, info: Dict[str, Any]):
        info.update({"invalid_move": True, "error": msg, "outcome": "invalid"})
        self.done = True
        return {"observation": f"Invalid: {msg}\nGame Over."}, self.invalid_penalty, True, info

    def _obs(self) -> str:
        facts = "\n  ".join(f"({p} {' '.join(a)})" for (p,a) in sorted(self.world.facts) if p != "adjacent")
        goals = " ".join(f"({g[0]} {' '.join(g[1])})" for g in self.goal)
        return f"State:\n  {facts}\nSteps: {self.steps}/{self.max_steps}\nGoal: {goals}\nReply with <move>(action args)</move>."
