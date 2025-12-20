"""
Task loader: resolves domain, realizes variants, renames atoms in S‑expressions,
hydrates WorldState, and returns a PDDLEnv + task metadata.
"""

from __future__ import annotations

import yaml
import re
import random
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Set, Optional, Dict, Any, List
from importlib.resources import files as pkg_files

from sere.pddl.domain_spec import DomainSpec
from sere.pddl.sexpr import parse_one, to_string, SExprError
from sere.core.world_state import WorldState
from sere.core.pddl_env import PDDLEnv


# =========================
#  Asset I/O helpers
# =========================

def _strip_leading(part: Path, prefix: str) -> Path:
    parts = part.parts
    if parts and parts[0] == prefix:
        return Path(*parts[1:])
    return part


def _load_yaml_from_task(task_path: str) -> Dict[str, Any]:
    """Load a task YAML from filesystem or from the packaged 'sere.assets.tasks'."""
    p = Path(task_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # package lookup (logical id like 'kitchen/foo.yaml' or 'tasks/kitchen/foo.yaml')
    rel = _strip_leading(Path(task_path), "tasks")
    cand = pkg_files("sere.assets.tasks") / str(rel)
    if cand.is_file():
        with cand.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(
        f"Task not found: {task_path} (looked on disk and under sere.assets.tasks/{rel})"
    )


def _load_domain_yaml(dom_path: str) -> Dict[str, Any]:
    """
    Load a domain YAML from filesystem or from the packaged 'sere.assets.domain'.
    """
    p = Path(dom_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # allow 'domain/...' or 'domains/...'
    rel = Path(dom_path)
    if rel.parts and rel.parts[0] in {"domain", "domains"}:
        rel = Path(*rel.parts[1:])
    cand = pkg_files("sere.assets.domain") / str(rel)
    if not cand.is_file():
        raise FileNotFoundError(
            f"Domain file not found: {dom_path} (looked on disk and under sere.assets.domain/{rel})"
        )
    with cand.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================
#  Env config / meta merge
# =========================


@dataclass
class EnvConfig:
    max_steps: int = 40
    step_penalty: float = -0.01
    invalid_penalty: float = -0.1
    enable_numeric: bool = False
    enable_conditional: bool = False
    enable_durations: bool = False
    enable_stochastic: bool = False
    time_limit: Optional[float] = None
    default_duration: float = 1.0
    seed: Optional[int] = None
    max_messages: int = 8
    reward_shaping: Optional[dict] = None
    multi_agent: bool = False

    @classmethod
    def from_meta(cls, meta: Dict[str, Any]) -> "EnvConfig":
        cfg = cls(
            max_steps=int(meta.get("max_steps", 40)),
            step_penalty=float(meta.get("step_penalty", -0.01)),
            invalid_penalty=float(meta.get("invalid_penalty", -0.1)),
            enable_numeric=bool(meta.get("enable_numeric", False)),
            enable_conditional=bool(meta.get("enable_conditional", False)),
            enable_durations=bool(meta.get("enable_durations", False)),
            enable_stochastic=bool(meta.get("enable_stochastic", False)),
            time_limit=meta.get("time_limit", None),
            default_duration=float(meta.get("default_duration", 1.0)),
            seed=meta.get("seed", None),
            max_messages=int(meta.get("max_messages", 8)),
            multi_agent=bool(meta.get("multi_agent", False)),
        )

        rs_cfg = meta.get("reward_shaping") or {}
        if rs_cfg:
            cfg.reward_shaping = dict(
                mode=(rs_cfg.get("mode") or "instant"),
                gamma=float(rs_cfg.get("gamma", 1.0)),
                milestones=[
                    dict(
                        expr=str(m.get("expr")),
                        reward=float(m.get("reward", 0.0)),
                        once=bool(m.get("once", True)),
                    )
                    for m in (rs_cfg.get("milestones") or [])
                    if m and m.get("expr") is not None
                ],
            )

        return cfg

    def apply_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        extras: Dict[str, Any] = {}
        for k, v in (overrides or {}).items():
            if hasattr(self, k):
                if k == "seed":
                    raise ValueError("Seed is resolved separately; do not override it here.")
                setattr(self, k, v)
            else:
                extras[k] = v
        return extras

    def to_env_kwargs(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("reward_shaping") is None:
            data.pop("reward_shaping", None)
        return data


@dataclass
class TaskSpec:
    id: str
    name: str
    description: str
    domain: str
    path: str
    meta: Dict[str, Any]
    objects: Dict[str, Set[str]]
    static_facts: Set[tuple]
    init_facts: Set[tuple]
    init_fluents: List[list]
    termination_rules: List[dict]
    reference_plan: List[str]
    realized_names: Dict[str, str]
    env_config: EnvConfig

    def to_task_meta(self, env: PDDLEnv) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "seed": self.env_config.seed,
            "features": {
                "numeric": env.enable_numeric,
                "conditional": env.enable_conditional,
                "durations": env.enable_durations,
                "stochastic": env.enable_stochastic,
                "multi_agent": env.multi_agent,
            },
            "limits": {"max_steps": env.max_steps, "time_limit": env.time_limit},
            "reference_plan": self.reference_plan,
            "path": self.path,
            "domain": self.domain,
            "termination": self.termination_rules,
            "realized_names": self.realized_names,
        }


# =========================
#  Domain resolution
# =========================

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
    raise ValueError(
        "Cannot determine domain. Set meta.domain in the task YAML or pass domain_path explicitly."
    )


# =========================
#  Parsing primitives
# =========================

def _parse_lit(s: str) -> Tuple[str, Tuple[str, ...]]:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"Bad literal: {s!r}")
    parts = s[1:-1].split()
    if not parts:
        raise ValueError(f"Bad literal: {s!r}")
    head, args = parts[0], parts[1:]
    return head.lower(), tuple(args)


def _apply_init_fluents(world: WorldState, init_fluents: List[list]) -> None:
    for entry in init_fluents:
        if not (isinstance(entry, list) and len(entry) == 3):
            raise ValueError(
                f"Bad init_fluents entry (want [name, [args...], value]): {entry}"
            )
        fname, args, val = entry
        if not isinstance(args, list):
            raise ValueError(f"init_fluents args must be a list: {entry}")
        world.set_fluent(str(fname).lower(), tuple(str(a) for a in args), float(val))


# =========================
#  Object schema + variants
# =========================

def _normalize_objects_strict(objs: dict, domain: DomainSpec) -> tuple[Dict[str, Set[str]], Dict[str, List[str]]]:
    """
    Strictly accept the NEW schema only:
      objects:
        roleA: type | [type, ...] | {types:[...], variants:[...]}
        roleB: {types:[...]}                      # variants optional
    Returns:
      - types_by_sym: {placeholder -> set(types)}
      - variants_by_sym: {placeholder -> [variants]}  (may be empty)
    Raises if any type isn’t declared in the domain.
    """
    if not isinstance(objs, dict) or not objs:
        raise ValueError(
            "objects must be a non-empty mapping of role -> type(s) or {types, variants}."
        )

    valid_types = set(domain.types.keys())

    types_by_sym: Dict[str, Set[str]] = {}
    variants_by_sym: Dict[str, List[str]] = {}

    for sym, val in objs.items():
        if isinstance(val, str):
            t = str(val).lower()
            if t not in valid_types:
                raise ValueError(
                    f"objects[{sym}] = {val!r} is not a declared type. Valid: {sorted(valid_types)}"
                )
            types_by_sym.setdefault(sym, set()).add(t)

        elif isinstance(val, list):
            if not val or not all(isinstance(t, str) for t in val):
                raise ValueError(f"objects[{sym}] list must contain type strings.")
            lowered = [str(t).lower() for t in val]
            bad = [t for t in lowered if t not in valid_types]
            if bad:
                raise ValueError(
                    f"objects[{sym}] has unknown types {bad}. Valid: {sorted(valid_types)}"
                )
            for t in lowered:
                types_by_sym.setdefault(sym, set()).add(t)

        elif isinstance(val, dict):
            ts = val.get("types")
            if ts is None:
                raise ValueError(f"objects[{sym}] dict must contain 'types'.")
            if isinstance(ts, str):
                ts = [ts]
            if not (isinstance(ts, list) and ts and all(isinstance(t, str) for t in ts)):
                raise ValueError(
                    f"objects[{sym}].types must be a non-empty string or list of strings."
                )
            lowered = [str(t).lower() for t in ts]
            bad = [t for t in lowered if t not in valid_types]
            if bad:
                raise ValueError(
                    f"objects[{sym}].types has unknown types {bad}. Valid: {sorted(valid_types)}"
                )
            types_by_sym[sym] = set(lowered)

            vs = val.get("variants") or []
            if not isinstance(vs, list) or (vs and not all(isinstance(x, str) for x in vs)):
                raise ValueError(
                    f"objects[{sym}].variants must be a list of strings if provided."
                )
            if vs:
                variants_by_sym[sym] = vs

        else:
            raise ValueError(f"objects[{sym}] must be str | list[str] | dict.")

    return types_by_sym, variants_by_sym


def _choose_variant_names_strict(
    placeholders: List[str],
    variants_by_sym: Dict[str, List[str]],
    rng: random.Random,
    protect: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """
    For each placeholder that has a 'variants' pool, pick a concrete name.
    Enforce global uniqueness across chosen names and against `protect`.
    Raise ValueError on collision or if a pool is empty.
    """
    used: Set[str] = set(protect or set())
    mapping: Dict[str, str] = {}

    for ph in placeholders:
        pool = variants_by_sym.get(ph, [])
        if not pool:
            continue  # placeholder without variants → stays as-is
        # Filter out protected collisions; if nothing left, raise.
        avail = [v for v in pool if v not in used]
        if not avail:
            raise ValueError(
                f"No available variant for placeholder '{ph}'. Pool={pool}, already used={sorted(used)}"
            )
        chosen = rng.choice(avail)
        if chosen in used:
            # Shouldn't happen given filtering; keep the guard.
            raise ValueError(f"Variant collision: '{chosen}' already used.")
        mapping[ph] = chosen
        used.add(chosen)

    return mapping

def _apply_clutter(
    meta: Dict[str, Any],
    world: WorldState,
    static_facts: Set[tuple],
    rng: random.Random,
):
    """
    Inject 'clutter' objects using counts-only sampling, then clamp to a global budget.

    Schema (counts-only):
      meta.clutter:
        budget: {min: int, max: int}    # total items after clamping (defaults: [0, +inf))
        items:
          - name: str
            types: [str, ...]           # domain-declared types
            variants: [str, ...]        # optional pool of unique names (mutually exclusive with auto-names)
            at: [location, ...]         # candidate locations (required)
            count: {min:int, max:int}   # required; sample N ~ Uniform[min, max] (inclusive)

    Behavior:
      1) For each item spec, sample a desired count N_i in [min,max].
      2) Compute total T = sum_i N_i.
      3) If T > budget.max, deterministically downsample across items until sum == budget.max.
      4) If T < budget.min, best-effort top-up by adding more of the earliest viable items until sum == budget.min.
      5) Spawn the final counts, picking unique names (variants first, else auto base_1, base_2, ...),
         and placing each at a random candidate location.
    """
    cl = (meta or {}).get("clutter") or {}
    items = cl.get("items") or []
    if not items:
        return

    # --- read global budget
    bmin = int(cl.get("budget", {}).get("min", 0))
    bmax = cl.get("budget", {}).get("max", None)
    bmax = int(bmax) if bmax is not None else 10**9
    if bmax < bmin:
        bmax = bmin

    # --- validate & sample desired counts per item
    desired: list[dict] = []   # [{'spec': spec, 'n': int}, ...] in same order
    for spec in items:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name") or "clutter")
        types = spec.get("types") or []
        locs  = spec.get("at") or []
        cnt   = spec.get("count") or {}
        if not types or not locs or "min" not in cnt or "max" not in cnt:
            continue
        cmin = int(cnt.get("min", 0))
        cmax = int(cnt.get("max", cmin))
        if cmax < cmin:
            cmax = cmin
        n = rng.randint(cmin, cmax) if cmax > cmin else cmin
        desired.append({"spec": spec, "n": n})

    # total desired
    total = sum(x["n"] for x in desired)

    # --- clamp DOWN to bmax if needed (deterministic/fair reduction)
    if total > bmax:
        # Make a working list of indices replicated n times, then shuffle and trim.
        # This spreads reductions fairly across items.
        slots: list[int] = []
        for i, x in enumerate(desired):
            slots.extend([i] * x["n"])
        rng.shuffle(slots)
        # Keep exactly bmax
        keep = slots[:bmax]
        # Recount per-item
        new_counts = [0] * len(desired)
        for i in keep:
            new_counts[i] += 1
        for i, x in enumerate(desired):
            x["n"] = new_counts[i]
        total = bmax

    # --- top-up to bmin if needed (best-effort round-robin)
    if total < bmin and desired:
        deficit = bmin - total
        # Round-robin over items; try to add while respecting each item's max (if variants may exhaust, spawn will guard)
        i = 0
        L = len(desired)
        while deficit > 0 and L > 0:
            x = desired[i]
            cnt = x["spec"].get("count") or {}
            cmax = int(cnt.get("max", x["n"]))
            if x["n"] < cmax:
                x["n"] += 1
                deficit -= 1
            i = (i + 1) % L
        total = bmin - max(deficit, 0)

    # --- spawn
    used = set(world.objects.keys())

    def _unique(base: str) -> str:
        i = 1
        cand = f"{base}_{i}"
        while cand in used:
            i += 1
            cand = f"{base}_{i}"
        used.add(cand)
        return cand

    for x in desired:
        spec = x["spec"]; n = x["n"]
        if n <= 0:
            continue
        base = str(spec.get("name") or "clutter")
        types = [str(t).lower() for t in (spec.get("types") or [])]
        locs  = list(spec.get("at") or [])
        variants = list(spec.get("variants") or [])  # optional

        for _ in range(n):
            # choose name
            if variants:
                avail = [v for v in variants if v not in used]
                if avail:
                    name = rng.choice(avail)
                    used.add(name)
                else:
                    # variant pool exhausted; fall back to auto-name
                    name = _unique(base)
            else:
                name = _unique(base)

            # register object + place
            for t in types:
                world.add_object(name, t)
            loc = str(rng.choice(locs))
            world.facts.add(("obj-at", (name, loc)))


# =========================
#  Rename utilities
# =========================

def _rename_atoms_nonheads_in_sexpr(s: str, mapping: Dict[str, str]) -> str:
    """
    Rename object symbols inside any S-expression, but NEVER rename head symbols.
    Works for nested forms: (and (p a) (q b)) and numeric: (>= (f x) 10)
    Strategy: parse and rename only non-head atoms (skip ?vars and numeric atoms).
    """
    if not isinstance(s, str):
        return s
    try:
        parsed = parse_one(s)
    except SExprError:
        return s

    def _rename(node):
        if isinstance(node, list):
            if not node:
                return node
            head = node[0]
            rest = [_rename(x) for x in node[1:]]
            return [head] + rest
        tok = str(node)
        if not tok:
            return tok
        if tok[0] == "?":  # variables untouched
            return tok
        if tok[0].isdigit():  # numbers untouched
            return tok
        return mapping.get(tok, tok)

    return to_string(_rename(parsed))


def _tokenwise_replace_lit(s: str, mapping: Dict[str, str]) -> str:
    """Replace whole tokens inside a single S-expr string (literal only)."""
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s
    try:
        parsed = parse_one(s)
    except SExprError:
        return s
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], str):
        return s
    if any(isinstance(x, list) for x in parsed[1:]):
        return s
    head = str(parsed[0])
    args = [mapping.get(str(t), str(t)) for t in parsed[1:]]
    return to_string([head] + args)


def _rename_everywhere(task: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    y = copy.deepcopy(task)

    # 1) objects: rename KEYS, strip variants in finalized form
    new_objs: Dict[str, Any] = {}
    for sym, val in (y.get("objects") or {}).items():
        new_key = mapping.get(sym, sym)
        if new_key in new_objs and new_key != sym:
            raise ValueError(f"Object name collision after mapping: {sym} -> {new_key}")
        if isinstance(val, dict):
            val = {k: v for k, v in val.items() if k != "variants"}
        new_objs[new_key] = val
    y["objects"] = new_objs

    # 2) S-expr arrays with single literals → keep literal replacer
    for key in ("init", "static_facts", "reference_plan"):
        if isinstance(y.get(key), list):
            y[key] = [_tokenwise_replace_lit(s, mapping) for s in y[key]]

    # 3) termination.when may be a complex clause; rename non-head atoms everywhere
    if isinstance(y.get("termination"), list):
        def _rename_term_block(block):
            if isinstance(block, str):
                return _rename_atoms_nonheads_in_sexpr(block, mapping)
            if isinstance(block, list):
                return [_rename_term_block(x) for x in block]
            if isinstance(block, dict):
                out = dict(block)
                if "when" in out:
                    out["when"] = _rename_term_block(out.get("when"))
                if "all" in out:
                    out["all"] = _rename_term_block(out.get("all"))
                if "any" in out:
                    out["any"] = _rename_term_block(out.get("any"))
                return out
            return block

        y["termination"] = [_rename_term_block(r) for r in y["termination"]]

    # 4) meta.init_fluents (args)
    mf = ((y.get("meta") or {}).get("init_fluents") or [])
    out: List[list] = []
    for entry in mf:
        if isinstance(entry, list) and len(entry) == 3:
            fname, args, val = entry
            args = [mapping.get(a, a) for a in args]
            out.append([fname, args, val])
        else:
            out.append(entry)
    if out:
        y.setdefault("meta", {})["init_fluents"] = out

    # 5) reward shaping milestone expressions may be full clauses; fix them too
    rs = (y.get("meta") or {}).get("reward_shaping") or {}
    ms = rs.get("milestones") or []
    changed = False
    new_ms: List[dict] = []
    for m in ms:
        if not isinstance(m, dict):
            new_ms.append(m)
            continue
        expr = m.get("expr")
        if isinstance(expr, str):
            m = dict(m)
            m["expr"] = _rename_atoms_nonheads_in_sexpr(expr, mapping)
            changed = True
        new_ms.append(m)
    if changed:
        y.setdefault("meta", {}).setdefault("reward_shaping", {})["milestones"] = new_ms

    return y


# =========================
#  Termination parsing
# =========================

def _combine_expr(op: str, exprs: List[str]) -> str:
    exprs = [e for e in (exprs or []) if isinstance(e, str) and e.strip()]
    if not exprs:
        raise ValueError(f"Termination '{op}' requires at least one clause.")
    if len(exprs) == 1:
        return exprs[0]
    return f"({op} {' '.join(exprs)})"


def _expr_from_block(block) -> str:
    if isinstance(block, str):
        s = block.strip()
        if not s:
            raise ValueError("Termination clause must be a non-empty string.")
        return s
    if isinstance(block, list):
        return _combine_expr("and", [_expr_from_block(x) for x in block])
    if isinstance(block, dict):
        if "when" in block:
            return _expr_from_block(block.get("when"))
        has_all = "all" in block
        has_any = "any" in block
        if has_all and has_any:
            raise ValueError("Termination block cannot contain both 'all' and 'any'.")
        if has_all:
            items = block.get("all")
            if not isinstance(items, list):
                raise ValueError("Termination 'all' must be a list.")
            return _combine_expr("and", [_expr_from_block(x) for x in items])
        if has_any:
            items = block.get("any")
            if not isinstance(items, list):
                raise ValueError("Termination 'any' must be a list.")
            return _combine_expr("or", [_expr_from_block(x) for x in items])
    raise ValueError(f"Bad termination clause: {block!r}")


def _parse_termination(yaml_block) -> List[dict]:
    rules = []
    for r in (yaml_block or []):
        if not isinstance(r, dict):
            raise ValueError(f"Termination rule must be a mapping: {r!r}")
        if "when" not in r and "all" not in r and "any" not in r:
            raise ValueError(f"Termination rule missing 'when'/'all'/'any': {r!r}")
        when_expr = _expr_from_block(r.get("when", r))
        rules.append(
            dict(
                name=str(r.get("name", "term")),
                when=when_expr,
                outcome=str(r.get("outcome", "terminal")),
                reward=float(r.get("reward", 0.0)),
            )
        )
    return rules


# =========================
#  Plugins
# =========================

def _load_invariants_plugin(domain_name: str) -> List[Any]:
    """Tries to load core.invariants.{DomainName}Invariants by naming convention."""
    try:
        from ..core import invariants as invmod
    except Exception:
        return []
    cname = f"{domain_name.capitalize()}Invariants"
    pl = getattr(invmod, cname, None)
    return [pl()] if pl else []


# =========================
#  Validation helpers (optional but recommended)
# =========================

# Symbols we treat as non-object heads/ops across expressions
_BUILTIN_HEADS = {
    "and", "or", "not", "distinct", "<", ">", "<=", ">=", "=",
}


def _symbols_in_sexpr(s: str) -> Set[str]:
    """Collect atom-like tokens; does not parse, best-effort for validation."""
    toks = set(re.findall(r"[A-Za-z0-9_-]+", s or ""))
    # Keep words that aren't obviously numeric
    return {t for t in toks if not t[0].isdigit()}


def _validate_expr_symbols_exist(
    expr: str,
    *,
    declared_objects: Set[str],
    domain: DomainSpec,
    where: str,
) -> None:
    """
    Best-effort guardrail: ensure that any atoms present in an expression are either
    declared objects (post-rename), predicate/ fluent names, or known operators.
    """
    if not isinstance(expr, str):
        return
    toks = _symbols_in_sexpr(expr)
    known_preds = set(domain.predicates.keys())
    known_fl = set(domain.fluents.keys())
    unknown: Set[str] = set()
    for tok in toks:
        if tok in declared_objects:
            continue
        tl = tok.lower()
        if tl in known_preds or tl in known_fl or tl in _BUILTIN_HEADS:
            continue
        unknown.add(tok)
    if unknown:
        # Don't false-positive on obvious location/var words if user forgot to declare.
        raise ValueError(f"Unknown symbols in {where}: {sorted(unknown)}")


# =========================
#  Main loader
# =========================

def load_task(domain_path: Optional[str], task_path: str, plugins=None, **env_kwargs) -> Tuple[PDDLEnv, dict]:
    y = _load_yaml_from_task(task_path)

    # ---- Domain hint BEFORE rename is fine
    meta = y.get("meta", {}) or {}
    overrides = dict(env_kwargs or {})
    seed_override = overrides.pop("seed", None)
    meta_seed = meta.get("seed", None)
    if seed_override is not None and meta_seed is not None:
        raise ValueError("Seed specified in both task meta and load_task; choose one source.")
    resolved_seed = seed_override if seed_override is not None else meta_seed
    domain_hint = (meta.get("domain") or "").strip().lower() or None
    dom_path = _resolve_domain_path(domain_hint, task_path, domain_path)
    dom_yaml = _load_domain_yaml(dom_path)
    dom = DomainSpec.from_dict(dom_yaml)

    # ---- Plugins
    if plugins is None:
        dom_stem = Path(dom_path).stem
        dn = domain_hint or _infer_domain_from_path(task_path) or dom_stem
        plugins = _load_invariants_plugin(dn)

    # ---- RNG
    rng = random.Random(resolved_seed)

    # ---- Objects (STRICT) + variant realization
    types_by_placeholder, variants_by_placeholder = _normalize_objects_strict(
        y.get("objects") or {}, dom
    )

    protected_names: Set[str] = {
        sym for sym in types_by_placeholder if sym not in variants_by_placeholder
    }

    mapping = _choose_variant_names_strict(
        placeholders=list(variants_by_placeholder.keys()),
        variants_by_sym=variants_by_placeholder,
        rng=rng,
        protect=protected_names,
    )

    # ---- APPLY RENAME FIRST
    if mapping:
        y = _rename_everywhere(y, mapping)
        # Recompute types AFTER rename
        types_by_placeholder, _ = _normalize_objects_strict(y.get("objects") or {}, dom)

    # re-read meta & termination AFTER rename
    meta = y.get("meta", {}) or {}
    termination_rules = _parse_termination(y.get("termination"))

    # ---- Validate expressions refer to declared objects (post-rename)
    declared_objects = set(types_by_placeholder.keys())
    for i, r in enumerate(termination_rules):
        _validate_expr_symbols_exist(
            r.get("when", ""),
            declared_objects=declared_objects,
            domain=dom,
            where=f"termination[{i}].when",
        )
    rs_cfg_tmp = (y.get("meta", {}) or {}).get("reward_shaping") or {}
    for j, m in enumerate(rs_cfg_tmp.get("milestones", []) or []):
        if isinstance(m, dict) and isinstance(m.get("expr"), str):
            _validate_expr_symbols_exist(
                m["expr"],
                declared_objects=declared_objects,
                domain=dom,
                where=f"reward_shaping.milestones[{j}].expr",
            )

    # --- Env config (merge meta + overrides once)
    cfg = EnvConfig.from_meta(meta)
    if resolved_seed is not None:
        cfg.seed = resolved_seed
    extras = cfg.apply_overrides(overrides)
    env_cfg = cfg.to_env_kwargs()
    env_cfg.update(extras)

    domain_name = domain_hint or Path(dom_path).stem
    spec = TaskSpec(
        id=y["id"],
        name=y.get("name", y["id"]),
        description=y.get("description", ""),
        domain=domain_name,
        path=task_path,
        meta=meta,
        objects=types_by_placeholder,
        static_facts=set(_parse_lit(x) for x in (y.get("static_facts") or [])),
        init_facts=set(_parse_lit(x) for x in (y.get("init") or [])),
        init_fluents=(meta.get("init_fluents") or []),
        termination_rules=termination_rules,
        reference_plan=y.get("reference_plan", []),
        realized_names=mapping,
        env_config=cfg,
    )

    # --- world & objects ---
    w = WorldState(dom)
    for sym, tys in spec.objects.items():
        for t in tys:
            w.add_object(sym, t)

    # --- facts
    static_facts = spec.static_facts
    for fact in spec.init_facts:
        w.facts.add(fact)

    # --- fluents (now renamed)
    if spec.init_fluents:
        _apply_init_fluents(w, spec.init_fluents)

    # --- Optional clutter injection ---
    try:
        _apply_clutter(spec.meta, w, static_facts, rng)
    except Exception:
        pass

    env = PDDLEnv(
        dom,
        w,
        static_facts,
        plugins=plugins or [],
        termination_rules=termination_rules,  # <-- renamed & validated version
        **env_cfg,
    )

    return env, spec.to_task_meta(env)
