import yaml, re
from pathlib import Path
from typing import Tuple, Set, Optional, Dict, Any, List
from importlib.resources import files as pkg_files, as_file

from sere.pddl.domain_spec import DomainSpec
from sere.core.world_state import WorldState
from sere.core.pddl_env import PDDLEnv

def _parse_lit(s: str):
    s = s.strip()
    assert s.startswith("(") and s.endswith(")"), f"Bad literal: {s}"
    toks = s[1:-1].split()
    return toks[0], tuple(toks[1:])

def _apply_init_fluents(world: WorldState, init_fluents: List[list]):
    for entry in init_fluents:
        if not (isinstance(entry, list) and len(entry) == 3):
            raise ValueError(f"Bad init_fluents entry (want [name, [args...], value]): {entry}")
        fname, args, val = entry
        # Time is engine-owned; ignore any attempts to seed 'elapsed'
        if str(fname) == "elapsed":
            continue
        if not isinstance(args, list):
            raise ValueError(f"init_fluents args must be a list: {entry}")
        world.set_fluent(str(fname), tuple(str(a) for a in args), float(val))

def _infer_domain_from_path(task_path: str) -> Optional[str]:
    p = task_path.replace("\\", "/").lower()
    m = re.search(r"/assets/tasks/([^/]+)/", p)
    return m.group(1) if m else None

def _resolve_domain_path(domain_hint: Optional[str], task_path: str, domain_path: Optional[str]) -> str:
    # 1) explicit path wins
    if domain_path:
        return domain_path
    # 2) meta.domain
    if domain_hint:
        sanitized = re.sub(r"[^a-z0-9_\-]", "", domain_hint.lower())
        return f"domain/{sanitized}.yaml"
    # 3) path heuristic fallback
    inferred = _infer_domain_from_path(task_path)
    if inferred:
        return f"domain/{inferred}.yaml"
    raise ValueError("Cannot determine domain. Set meta.domain in the task YAML or pass domain_path explicitly.")

# --- tiny helpers to read from packaged assets when files aren't on disk ---

def _strip_leading(part: Path, prefix: str) -> Path:
    parts = part.parts
    if parts and parts[0] == prefix:
        return Path(*parts[1:])
    return part

def _load_yaml_from_task(task_path: str) -> Dict[str, Any]:
    """Try filesystem first; else load from package sere.assets.tasks (supports legacy 'tasks/...')."""
    p = Path(task_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # package lookup (logical id like 'kitchen/foo.yaml' or legacy 'tasks/kitchen/foo.yaml')
    rel = _strip_leading(Path(task_path), "tasks")
    cand = pkg_files("sere.assets.tasks") / str(rel)
    if cand.is_file():
        with cand.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Task not found: {task_path} (looked on disk and under sere.assets.tasks/{rel})")

def _resolve_domain_file(dom_path: str) -> Path:
    """
    Return a real filesystem Path to the domain YAML, resolving against
    packaged sere.assets.domain if not present on disk. Uses as_file() so
    DomainSpec.from_yaml can consume a path string.
    """
    p = Path(dom_path)
    if p.exists():
        return p
    # allow 'domain/...' or 'domains/...'
    rel = Path(dom_path)
    if rel.parts and rel.parts[0] in {"domain", "domains"}:
        rel = Path(*rel.parts[1:])
    cand = pkg_files("sere.assets.domain") / str(rel)
    if not cand.is_file():
        raise FileNotFoundError(f"Domain file not found: {dom_path} (looked on disk and under sere.assets.domain/{rel})")
    # materialize to a real path for downstream code
    with as_file(cand) as real_path:
        return Path(real_path)

def _load_invariants_plugin(domain_name: str):
    """Tries to load core.invariants.{DomainName}Invariants by naming convention."""
    try:
        from ..core import invariants as invmod
    except Exception:
        return []
    # Accept 'assembly' -> 'AssemblyInvariants', 'kitchen' -> 'KitchenInvariants'
    cname = f"{domain_name.capitalize()}Invariants"
    pl = getattr(invmod, cname, None)
    return [pl()] if pl else []

def load_task(
    domain_path: Optional[str],
    task_path: str,
    plugins=None,
    **env_kwargs
) -> Tuple[PDDLEnv, dict]:
    """
    Load a task YAML and construct a PDDLEnv using the convention:
      domain file at 'domain/{meta.domain}.yaml'
    Caller can still override via domain_path or plugins.
    """
    y = _load_yaml_from_task(task_path)

    meta = y.get("meta", {}) or {}

    domain_hint = (meta.get("domain") or "").strip().lower() or None
    dom_path = _resolve_domain_path(domain_hint, task_path, domain_path)

    dom_file = _resolve_domain_file(dom_path)

    dom = DomainSpec.from_yaml(str(dom_file))

    # Plugins: caller override > convention > none
    if plugins is None:
        dn = domain_hint or _infer_domain_from_path(task_path) or dom_file.stem
        plugins = _load_invariants_plugin(dn)

    # --- world & objects ---
    w = WorldState(dom)
    for typ, ids in y["objects"].items():
        for sym in ids:
            w.add_object(sym, typ)

    static_facts: Set[tuple] = set(_parse_lit(x) for x in y.get("static_facts", []))
    for fact in y.get("init", []):
        w.facts.add(_parse_lit(fact))
    goals = [_parse_lit(g) for g in y["goal"]]

    # Optional initial fluents
    init_fluents = meta.get("init_fluents", [])
    if init_fluents:
        _apply_init_fluents(w, init_fluents)

    # --- Env config (YAML meta < explicit kwargs) ---
    env_cfg = {
        "max_steps":           meta.get("max_steps", 40),
        "step_penalty":        meta.get("step_penalty", -0.01),
        "invalid_penalty":     meta.get("invalid_penalty", -0.1),
        "goal_reward":         meta.get("goal_reward", 1.0),
        "enable_numeric":      meta.get("enable_numeric", False),
        "enable_conditional":  meta.get("enable_conditional", False),
        "enable_durations":    meta.get("enable_durations", False),
        "enable_stochastic":   meta.get("enable_stochastic", False),
        "time_limit":          meta.get("time_limit", None),
        "default_duration":    meta.get("default_duration", 1.0),
        "seed":                meta.get("seed", None),
        "max_messages":        meta.get("max_messages", 8),
    }
    env_cfg.update(env_kwargs or {})

    env = PDDLEnv(dom, w, static_facts, goals, plugins=plugins or [], **env_cfg)

    task_meta = {
        "id": y["id"],
        "name": y.get("name", y["id"]),
        "description": y.get("description", ""),
        "seed": meta.get("seed", None),
        "features": {
            "numeric": env.enable_numeric,
            "conditional": env.enable_conditional,
            "durations": env.enable_durations,
            "stochastic": env.enable_stochastic,
        },
        "limits": {"max_steps": env.max_steps, "time_limit": env.time_limit},
        "reference_plan": y.get("reference_plan", []),
        "path": task_path,
        "domain": domain_hint or dom_file.stem,
    }
    return env, task_meta
