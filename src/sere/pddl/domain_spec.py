from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import yaml

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
    when: List[str]
    add: List[str]
    delete: List[str]
    num_eff: List[str]
    messages: List[str] = field(default_factory=list)

@dataclass
class PredicateSpec:
    name: str
    args: List[Tuple[str, str]]
    nl: List[str]                # variants
    static: bool = False

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

    @staticmethod
    def from_yaml(path: str) -> "DomainSpec":
        y = yaml.safe_load(open(path, "r"))

        # types
        types: Dict[str, str] = {}
        for t in y.get("types", []):
            raw = t["name"]
            if ":" in raw:
                a, b = [s.strip() for s in raw.split(":")]
                types[a] = b
            else:
                types[raw] = ""

        # predicates (nl -> List[str])
        preds: Dict[str, PredicateSpec] = {}
        for p in y.get("predicates", []):
            args = [(a["name"], a["type"]) for a in p.get("args", [])]
            preds[p["name"]] = PredicateSpec(
                name=p["name"],
                args=args,
                nl=_as_nl_list(p.get("nl"), p["name"]),
                static=p.get("static", False),
            )

        # fluents (nl -> List[str])
        fls: Dict[str, FluentSpec] = {}
        for f in y.get("fluents", []):
            args = [(a["name"], a["type"]) for a in f.get("args", [])]
            fls[f["name"]] = FluentSpec(
                name=f["name"],
                args=args,
                nl=_as_nl_list(f.get("nl"), f["name"]),
            )

        # actions (nl already normalized)
        actions: Dict[str, ActionSpec] = {}
        for a in y["actions"]:
            params: List[Tuple[str, str]] = []
            for d in a.get("params", []):
                ((var, typ),) = d.items()
                params.append((var, typ))

            # cond blocks
            cond_blocks: List[ConditionalBlock] = []
            for cb in a.get("cond", []) or []:
                cond_blocks.append(ConditionalBlock(
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
                    add=oc.get("add", []) or [],
                    delete=oc.get("del", oc.get("delete", [])) or [],
                    num_eff=oc.get("num_eff", []) or [],
                    when=oc.get("when", []) or [],
                    messages=oc.get("messages", []) or [],
                ))

            actions[a["name"]] = ActionSpec(
                name=a["name"],
                params=params,
                pre=a.get("pre", []) or [],
                add=a.get("add", []) or [],
                delete=a.get("del", a.get("delete", [])) or [],
                nl=_as_nl_list(a.get("nl"), a["name"]),
                num_eff=a.get("num_eff", []) or [],
                cond=cond_blocks or [],
                duration=a.get("duration", None),
                duration_var=a.get("duration_var"),
                duration_unit=a.get("duration_unit"),
                messages=a.get("messages", []) or [],
                outcomes=outcomes or [],
            )

        return DomainSpec(y["domain"], types, preds, actions, fls)
