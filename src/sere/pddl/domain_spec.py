from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from pathlib import Path
import yaml

from .sexpr import parse_one, SExprError
Predicate = Tuple[str, Tuple[str, ...]]

# ---------- helper: normalize nl -> List[str] ----------
def _as_nl_list(v: Any, fallback: str) -> List[str]:
    if v is None:
        return [fallback]
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else [fallback]
    if isinstance(v, (list, tuple)):
        out = [str(x).strip() for x in v if str(x).strip()]
        return out or [fallback]
    return [fallback]

@dataclass
class OutcomeSpec:
    name: str
    p: float
    status: Optional[str] = None
    add: List[str] = field(default_factory=list)
    delete: List[str] = field(default_factory=list)
    num_eff: List[str] = field(default_factory=list)
    when: List[str] = field(default_factory=list)      # optional guards
    messages: List[str] = field(default_factory=list)  # optional agent messages

@dataclass
class FluentSpec:
    name: str
    args: List[Tuple[str, str]]  # (var, type)
    nl: List[str]                # variants

@dataclass
class ConditionalBlock:
    forall: List[Tuple[str, str]] = field(default_factory=list)
    when: List[str] = field(default_factory=list)
    add: List[str] = field(default_factory=list)
    delete: List[str] = field(default_factory=list)
    num_eff: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

@dataclass
class PredicateSpec:
    name: str
    args: List[Tuple[str, str]]
    nl: List[str]                # variants
    static: bool = False

@dataclass
class DerivedRule:
    name: str
    head_terms: List[str]
    when: List[str]
    vars_in_when: Optional[Set[str]] = None
    constraints: Optional[Dict[str, Set[str]]] = None

@dataclass
class ActionSpec:
    name: str
    params: List[Tuple[str, str]]
    pre: List[str]                  # may include "(not ...)"
    add: List[str]
    delete: List[str]
    nl: List[str]                   # variants
    num_eff: List[str] = field(default_factory=list)
    cond: List[ConditionalBlock] = field(default_factory=list)
    duration: Optional[float] = None
    duration_var: str | None = None
    duration_unit: float | None = None
    messages: List[str] = field(default_factory=list)
    outcomes: List[OutcomeSpec] = field(default_factory=list)

@dataclass
class DomainSpec:
    name: str
    types: Dict[str, str]
    predicates: Dict[str, PredicateSpec]
    actions: Dict[str, ActionSpec]
    fluents: Dict[str, FluentSpec]
    derived: Dict[str, List[DerivedRule]]

    @staticmethod
    def from_yaml(path: str | Path) -> "DomainSpec":
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        return DomainSpec.from_dict(y)

    @staticmethod
    def from_dict(y: Dict[str, Any]) -> "DomainSpec":

        # types
        types: Dict[str, str] = {}
        for t in y.get("types", []):
            parent = ""
            if isinstance(t, str):
                raw = t
            elif isinstance(t, dict):
                raw = t.get("name")
                if raw is None:
                    raise ValueError("Type entry missing 'name'.")
                parent = str(t.get("parent", "") or "")
            else:
                raise ValueError(f"Bad type entry: {t!r}")
            if ":" in raw:
                a, b = [s.strip() for s in raw.split(":", 1)]
                name = a.strip().lower()
                parent = (parent or b).strip().lower()
                types[name] = parent
            else:
                name = str(raw).strip().lower()
                types[name] = parent.strip().lower() if parent else ""

        # predicates (nl -> List[str])
        preds: Dict[str, PredicateSpec] = {}
        for p in y.get("predicates", []):
            pname = str(p["name"]).lower()
            args = [(a["name"], str(a["type"]).lower()) for a in p.get("args", [])]
            preds[pname] = PredicateSpec(
                name=pname,
                args=args,
                nl=_as_nl_list(p.get("nl"), pname),
                static=p.get("static", False),
            )

        # fluents (nl -> List[str])
        fls: Dict[str, FluentSpec] = {}
        for f in y.get("fluents", []):
            fname = str(f["name"]).lower()
            args = [(a["name"], str(a["type"]).lower()) for a in f.get("args", [])]
            fls[fname] = FluentSpec(
                name=fname,
                args=args,
                nl=_as_nl_list(f.get("nl"), fname),
            )

        # actions (nl already normalized)
        actions: Dict[str, ActionSpec] = {}
        for a in y["actions"]:
            aname = str(a["name"]).lower()
            params: List[Tuple[str, str]] = []
            for d in a.get("params", []):
                ((var, typ),) = d.items()
                params.append((var, str(typ).lower()))

            def _parse_varspecs(raw) -> List[Tuple[str, str]]:
                out: List[Tuple[str, str]] = []
                for entry in raw or []:
                    if isinstance(entry, dict):
                        if "name" in entry and "type" in entry:
                            var = str(entry["name"]).lstrip("?")
                            typ = str(entry["type"]).lower()
                        else:
                            if len(entry) != 1:
                                raise ValueError(f"Bad quantifier var spec: {entry!r}")
                            ((var, typ),) = entry.items()
                            var = str(var).lstrip("?")
                            typ = str(typ).lower()
                    else:
                        raise ValueError(f"Bad quantifier var spec: {entry!r}")
                    out.append((var, typ))
                return out

            # cond blocks
            cond_blocks: List[ConditionalBlock] = []
            for cb in a.get("cond", []) or []:
                cond_blocks.append(ConditionalBlock(
                    forall=_parse_varspecs(cb.get("forall") or []),
                    when=cb.get("when", []) or [],
                    add=cb.get("add", []) or [],
                    delete=cb.get("del", cb.get("delete", [])) or [],
                    num_eff=cb.get("num_eff", []) or [],
                    messages=cb.get("messages", []) or [],
                ))

            # outcomes
            outcomes: List[OutcomeSpec] = []
            for oc in a.get("outcomes", []) or []:
                outcomes.append(OutcomeSpec(
                    name=oc.get("name", "outcome"),
                    p=float(oc["p"]),
                    status=(str(oc.get("status")) if oc.get("status") is not None else None),
                    add=oc.get("add", []) or [],
                    delete=oc.get("del", oc.get("delete", [])) or [],
                    num_eff=oc.get("num_eff", []) or [],
                    when=oc.get("when", []) or [],
                    messages=oc.get("messages", []) or [],
                ))

            actions[aname] = ActionSpec(
                name=aname,
                params=params,
                pre=a.get("pre", []) or [],
                add=a.get("add", []) or [],
                delete=a.get("del", a.get("delete", [])) or [],
                nl=_as_nl_list(a.get("nl"), aname),
                num_eff=a.get("num_eff", []) or [],
                cond=cond_blocks or [],
                duration=a.get("duration", None),
                duration_var=a.get("duration_var"),
                duration_unit=a.get("duration_unit"),
                messages=a.get("messages", []) or [],
                outcomes=outcomes or [],
            )

        _validate_static_effects(actions, preds)

        derived = _parse_derived(y.get("derived") or [], preds)

        return DomainSpec(y["domain"], types, preds, actions, fls, derived)

    def supertypes(self, typ: str) -> List[str]:
        """Return all ancestor types for `typ` (excluding `typ`), nearest-first."""
        out: List[str] = []
        seen = set()
        cur = str(typ).lower()
        while cur and cur not in seen:
            seen.add(cur)
            cur = self.types.get(cur, "") or ""
            if cur:
                out.append(cur)
        return out

    def is_subtype(self, child: str, parent: str) -> bool:
        """Return True if child == parent or child inherits from parent."""
        child = str(child).lower()
        parent = str(parent).lower()
        if child == parent:
            return True
        cur = child
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            cur = self.types.get(cur, "") or ""
            if cur == parent:
                return True
        return False


def _effect_head(expr: str) -> str:
    if not isinstance(expr, str):
        raise ValueError(f"Effect must be a string literal, got: {expr!r}")
    try:
        parsed = parse_one(expr)
    except SExprError as exc:
        raise ValueError(f"Bad effect literal: {expr!r}") from exc
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], str):
        raise ValueError(f"Bad effect literal: {expr!r}")
    head = str(parsed[0]).lower()
    if head != "not":
        return head
    if len(parsed) != 2 or not isinstance(parsed[1], list) or not parsed[1]:
        raise ValueError(f"Bad effect literal: {expr!r}")
    inner = parsed[1]
    if not isinstance(inner[0], str):
        raise ValueError(f"Bad effect literal: {expr!r}")
    return str(inner[0]).lower()


def _parse_derived(items: List[Dict[str, Any]], preds: Dict[str, PredicateSpec]) -> Dict[str, List[DerivedRule]]:
    rules: Dict[str, List[DerivedRule]] = {}
    if not items:
        return rules
    for d in items:
        if not isinstance(d, dict):
            raise ValueError(f"Bad derived entry: {d!r}")
        head = d.get("head")
        if not isinstance(head, str) or not head.strip():
            raise ValueError(f"Derived rule missing head: {d!r}")
        try:
            parsed = parse_one(head)
        except SExprError as exc:
            raise ValueError(f"Bad derived head: {head!r}") from exc
        if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], str):
            raise ValueError(f"Bad derived head: {head!r}")
        if any(isinstance(x, list) for x in parsed[1:]):
            raise ValueError(f"Bad derived head: {head!r}")
        name = str(parsed[0]).lower()
        if name not in preds:
            raise ValueError(f"Derived predicate '{name}' not declared in predicates.")
        head_terms = [str(x) for x in parsed[1:]]
        when = d.get("when")
        if isinstance(when, str):
            when_list = [when]
        elif isinstance(when, list):
            when_list = [str(x) for x in when if str(x).strip()]
        else:
            raise ValueError(f"Derived rule '{head}' must include 'when' as str or list.")
        if not when_list:
            raise ValueError(f"Derived rule '{head}' has empty 'when'.")
        rules.setdefault(name, []).append(DerivedRule(name=name, head_terms=head_terms, when=when_list))
    return rules


def _validate_static_effects(actions: Dict[str, ActionSpec], preds: Dict[str, PredicateSpec]) -> None:
    static_preds = {name for name, spec in (preds or {}).items() if spec.static}
    if not static_preds:
        return
    offenders: List[str] = []

    def _scan(effects: List[str], where: str) -> None:
        for eff in effects or []:
            pname = _effect_head(eff)
            if pname in static_preds:
                offenders.append(f"{where}: {eff}")

    for act in (actions or {}).values():
        _scan(act.add, f"{act.name}.add")
        _scan(act.delete, f"{act.name}.delete")
        for i, cb in enumerate(act.cond or []):
            _scan(cb.add, f"{act.name}.cond[{i}].add")
            _scan(cb.delete, f"{act.name}.cond[{i}].delete")
        for i, oc in enumerate(act.outcomes or []):
            _scan(oc.add, f"{act.name}.outcomes[{i}].add")
            _scan(oc.delete, f"{act.name}.outcomes[{i}].delete")

    if offenders:
        raise ValueError(
            "Static predicate effects are not allowed: " + "; ".join(offenders)
        )
