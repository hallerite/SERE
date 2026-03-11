"""
PDDL directory loader: loads standard .pddl domain + problem files
with an optional extensions.yaml overlay for SERE-specific features.

Directory layout:
    domain_dir/
        domain.pddl          # standard PDDL domain
        extensions.yaml       # optional: NL, outcomes, reward shaping, meta
        problems/             # optional subdirectory
            p01.pddl
            p02.pddl
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sere.pddl.domain_spec import DomainSpec
from sere.pddl.pddl_parser import (
    PDDLDomain,
    PDDLProblem,
    domain_to_spec,
    infer_static_predicates,
    parse_domain_file,
    parse_problem_file,
)
from sere.core.world_state import WorldState
from sere.core.pddl_env import PDDLEnv
from .task_loader import EnvConfig, _apply_init_fluents, _load_invariants_plugin, _parse_termination


# ---------------------------------------------------------------------------
#  Load domain + extensions
# ---------------------------------------------------------------------------

def load_pddl_domain(
    domain_dir: str | Path,
    *,
    domain_file: str = "domain.pddl",
    extensions_file: str = "extensions.yaml",
) -> Tuple[DomainSpec, Dict[str, Any], PDDLDomain]:
    """
    Load a PDDL domain directory.

    Returns (DomainSpec, extensions_meta, raw_PDDLDomain).
    """
    domain_dir = Path(domain_dir)
    pddl_path = domain_dir / domain_file
    if not pddl_path.exists():
        raise FileNotFoundError(f"Domain file not found: {pddl_path}")

    pddl_domain = parse_domain_file(pddl_path)

    # Load optional extensions
    ext_path = domain_dir / extensions_file
    extensions: Dict[str, Any] = {}
    if ext_path.exists():
        with open(ext_path, "r", encoding="utf-8") as f:
            extensions = yaml.safe_load(f) or {}

    # Build NL overrides dict
    nl_overrides = _build_nl_overrides(extensions)

    # Static overrides: combine inferred + explicit
    static_overrides = _compute_statics(pddl_domain, extensions)

    # Outcome overrides
    outcome_overrides = _extract_outcome_overrides(extensions)

    spec = domain_to_spec(
        pddl_domain,
        nl_overrides=nl_overrides,
        static_overrides=static_overrides,
        outcome_overrides=outcome_overrides,
    )

    meta = extensions.get("meta", {}) or {}
    return spec, meta, pddl_domain


def _build_nl_overrides(extensions: Dict[str, Any]) -> Dict[str, Any]:
    """Build the nl_overrides dict from extensions.yaml."""
    return {
        "predicates": extensions.get("predicates", {}),
        "actions": extensions.get("actions", {}),
        "fluents": extensions.get("fluents", {}),
        "derived": extensions.get("derived", []),
    }


def _compute_statics(
    pddl_domain: PDDLDomain, extensions: Dict[str, Any]
) -> Set[str]:
    """Compute static predicates from inference + explicit overrides + outcome scanning."""
    from sere.pddl.sexpr import parse_one as _parse_one, SExprError as _SExprError

    statics = infer_static_predicates(pddl_domain)

    # Also remove predicates modified by extension outcomes
    for _aname, ainfo in extensions.get("actions", {}).items():
        if not isinstance(ainfo, dict):
            continue
        raw_ocs = ainfo.get("outcomes", []) or []
        # outcomes can be a list of dicts or a dict of name->info (NL-only format)
        oc_list = raw_ocs if isinstance(raw_ocs, list) else []
        for oc in oc_list:
            if not isinstance(oc, dict):
                continue
            for eff in (oc.get("add", []) or []) + (oc.get("delete", oc.get("del", [])) or []):
                try:
                    p = _parse_one(eff)
                    if isinstance(p, list) and p:
                        statics.discard(str(p[0]).lower())
                except _SExprError:
                    pass

    # Apply explicit overrides
    for pname, pinfo in extensions.get("predicates", {}).items():
        if isinstance(pinfo, dict):
            if pinfo.get("static") is True:
                statics.add(pname.lower())
            elif pinfo.get("static") is False:
                statics.discard(pname.lower())
    return statics


def _extract_outcome_overrides(extensions: Dict[str, Any]) -> Dict[str, list]:
    """Extract per-action outcome definitions from extensions."""
    result: Dict[str, list] = {}
    for aname, ainfo in extensions.get("actions", {}).items():
        if isinstance(ainfo, dict) and "outcomes" in ainfo:
            result[aname.lower()] = ainfo["outcomes"]
    return result


# ---------------------------------------------------------------------------
#  Load problem as SERE task
# ---------------------------------------------------------------------------

def load_pddl_task(
    domain_dir: str | Path,
    problem_path: str | Path,
    *,
    plugins: Optional[list] = None,
    **env_kwargs,
) -> Tuple[PDDLEnv, dict]:
    """
    Load a PDDL problem as a SERE task.

    Returns the same (PDDLEnv, task_meta) tuple as task_loader.load_task.
    """
    domain_dir = Path(domain_dir)
    problem_path = Path(problem_path)

    # Load domain
    dom, extensions_meta, pddl_domain = load_pddl_domain(domain_dir)

    # Parse problem
    problem = parse_problem_file(problem_path)

    # Resolve env config
    meta = dict(extensions_meta)
    overrides = dict(env_kwargs or {})
    seed_override = overrides.pop("seed", None)

    cfg = EnvConfig.from_meta(meta)
    if seed_override is not None:
        cfg.seed = seed_override

    extras = cfg.apply_overrides(overrides)
    env_cfg = cfg.to_env_kwargs()
    env_cfg.update(extras)

    # Build world state
    w = WorldState(dom)

    # Register constants from domain
    for const_name, const_type in pddl_domain.constants:
        w.add_object(const_name, const_type)

    # Register objects from problem
    for obj_name, obj_type in problem.objects:
        w.add_object(obj_name, obj_type)

    # Separate init facts into static and dynamic
    static_preds = {name for name, spec in dom.predicates.items() if spec.static}
    static_facts: Set[tuple] = set()
    for pred, args in problem.init_facts:
        if pred in static_preds:
            static_facts.add((pred, args))
        else:
            w.facts.add((pred, args))

    # Init fluents
    if problem.init_fluents:
        for fname, fargs, fval in problem.init_fluents:
            w.set_fluent(fname, fargs, fval)

    # Termination rules
    ext_termination = meta.get("termination")
    if ext_termination:
        termination_rules = _parse_termination(ext_termination)
    else:
        # Default: PDDL goal -> success termination
        termination_rules = [
            dict(name="goal", when=problem.goal, outcome="success", reward=1.0),
        ]

    # Plugins
    if plugins is None:
        dn = dom.name.lower().replace(" ", "_")
        plugins = _load_invariants_plugin(dn)

    # Default: disable affordance listing for PDDL domains (avoids
    # expensive grounding).  Extensions or env_kwargs can override.
    fmt_cfg = env_cfg.pop("formatter_config", {})
    fmt_cfg.setdefault("show_affordances", False)

    env = PDDLEnv(
        dom,
        w,
        static_facts,
        plugins=plugins or [],
        termination_rules=termination_rules,
        formatter_config=fmt_cfg,
        **env_cfg,
    )

    task_meta = {
        "id": problem.name or problem_path.stem,
        "name": problem.name or problem_path.stem,
        "description": f"PDDL problem: {problem_path.name}",
        "seed": cfg.seed,
        "features": {
            "numeric": env.enable_numeric,
            "conditional": env.enable_conditional,
            "durations": env.enable_durations,
            "stochastic": env.enable_stochastic,
            "multi_agent": env.multi_agent,
        },
        "limits": {"max_steps": env.max_steps, "time_limit": env.time_limit},
        "reference_plan": [],
        "path": str(problem_path),
        "domain": dom.name,
        "termination": termination_rules,
        "realized_names": {},
    }

    return env, task_meta
