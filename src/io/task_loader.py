import yaml
from typing import Tuple, Set, Optional, Dict, Any, List
from ..pddl.domain_spec import DomainSpec
from ..core.world_state import WorldState
from ..core.pddl_env import PDDLEnv

def _parse_lit(s: str):
    s = s.strip()
    assert s.startswith("(") and s.endswith(")"), f"Bad literal: {s}"
    toks = s[1:-1].split()
    return toks[0], tuple(toks[1:])

def _apply_init_fluents(world: WorldState, init_fluents: List[list]):
    """
    Each entry is ["fname", [arg1, arg2, ...], value]
    Args list can be empty for zero-arity fluents.
    """
    for entry in init_fluents:
        if not (isinstance(entry, list) and len(entry) == 3):
            raise ValueError(f"Bad init_fluents entry (want [name, [args...], value]): {entry}")
        fname, args, val = entry
        if not isinstance(args, list):
            raise ValueError(f"init_fluents args must be a list: {entry}")
        world.set_fluent(str(fname), tuple(str(a) for a in args), float(val))

def load_task(
    domain_path: str,
    task_path: str,
    plugins=None,
    **env_kwargs
) -> Tuple[PDDLEnv, dict]:
    """
    Loads a task YAML and constructs a PDDLEnv.
    Priority of env settings:
      explicit kwargs in caller > meta in YAML > PDDLEnv defaults.
    """
    dom = DomainSpec.from_yaml(domain_path)
    y = yaml.safe_load(open(task_path, "r"))

    # --- world & objects ---
    w = WorldState(dom)
    for typ, ids in y["objects"].items():
        for sym in ids:
            w.add_object(sym, typ)

    static_facts: Set[tuple] = set(_parse_lit(x) for x in y.get("static_facts", []))
    for fact in y.get("init", []):
        w.facts.add(_parse_lit(fact))
    goals = [_parse_lit(g) for g in y["goal"]]

    # --- meta (optional) ---
    meta = y.get("meta", {}) or {}
    meta_seed = meta.get("seed", None)

    # Optional initial fluents
    init_fluents = meta.get("init_fluents", [])
    if init_fluents:
        _apply_init_fluents(w, init_fluents)

    # Compose env settings (YAML meta can be overridden by explicit env_kwargs)
    env_cfg = {
        "max_steps":           meta.get("max_steps", 40),
        "step_penalty":        meta.get("step_penalty", -0.01),
        "invalid_penalty":     meta.get("invalid_penalty", -0.1),
        "goal_reward":         meta.get("goal_reward", 1.0),
        "enable_numeric":      meta.get("enable_numeric", False),
        "enable_conditional":  meta.get("enable_conditional", False),
        "enable_durations":    meta.get("enable_durations", False),
        "time_limit":          meta.get("time_limit", None),
        "default_duration":    meta.get("default_duration", 1.0),
        "visible_fluents":     meta.get("visible_fluents", ["*"]),
        # Important: we keep the constructor seed; you can also reseed on reset(seed=...)
        "seed":                meta_seed,
    }
    # Allow caller to override anything
    env_cfg.update(env_kwargs or {})

    env = PDDLEnv(
        dom, w, static_facts, goals, plugins=plugins or [], **env_cfg
    )

    # Human metadata to return alongside the env
    task_meta = {
        "id": y["id"],
        "name": y.get("name", y["id"]),
        "description": y.get("description", ""),
        "seed": meta.get("seed", None),
        "features": {
            "numeric": env.enable_numeric,
            "conditional": env.enable_conditional,
            "durations": env.enable_durations,
        },
        "limits": {
            "max_steps": env.max_steps,
            "time_limit": env.time_limit,
        },
        # for testing solvability
        "reference_plan": y.get("reference_plan", []),
        "path": task_path,
    }
    return env, task_meta
