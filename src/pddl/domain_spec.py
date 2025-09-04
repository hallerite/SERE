from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import yaml

Predicate = Tuple[str, Tuple[str, ...]]

@dataclass
class FluentSpec:
    name: str
    args: List[Tuple[str, str]]  # (var, type)
    nl: str

@dataclass
class ConditionalBlock:
    when: List[str]              # e.g., ["(open ?k)", "(not (open ?m))", "(>= (water-temp ?k) 80)"]
    add: List[str]
    delete: List[str]
    num_eff: List[str]           # e.g., ["(increase (elapsed) 1)"]

@dataclass
class PredicateSpec:
    name: str
    args: List[Tuple[str, str]]
    nl: str
    static: bool = False

@dataclass
class ActionSpec:
    name: str
    params: List[Tuple[str, str]]
    pre: List[str]                  # may include "(not ...)"
    add: List[str]
    delete: List[str]
    nl: str
    num_pre: List[str] = []       # numeric guards, e.g. "(>= (energy ?r) 1)"
    num_eff: List[str] = []       # numeric effects "(increase (elapsed) 1)"
    cond: List[ConditionalBlock] = []
    duration: Optional[float] = 0

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
        types = {}
        for t in y.get("types", []):
            raw = t["name"]
            if ":" in raw:
                a, b = [s.strip() for s in raw.split(":")]
                types[a] = b
            else:
                types[raw] = ""
        # predicates
        preds = {}
        for p in y.get("predicates", []):
            args = [(a["name"], a["type"]) for a in p.get("args", [])]
            preds[p["name"]] = PredicateSpec(
                name=p["name"], args=args,
                nl=p.get("nl", p["name"]),
                static=p.get("static", False)
            )
        # --- fluents ---
        fls = {}
        for f in y.get("fluents", []):
            args = [(a["name"], a["type"]) for a in f.get("args", [])]
            fls[f["name"]] = FluentSpec(
                name=f["name"], args=args, nl=f.get("nl", f["name"])
            )
        # actions
        actions = {}
        for a in y["actions"]:
            params = []
            for d in a.get("params", []):
                ((var, typ),) = d.items()
                params.append((var, typ))
            # --- cond blocks ---
            cond_blocks = []
            for cb in a.get("cond", []) or []:
                cond_blocks.append(ConditionalBlock(
                    when=cb.get("when", []) or [],
                    add=cb.get("add", []) or [],
                    delete=cb.get("del", cb.get("delete", [])) or [],
                    num_eff=cb.get("num_eff", []) or []
                ))
            actions[a["name"]] = ActionSpec(
                name=a["name"],
                params=params,
                pre=a.get("pre", []) or [],
                add=a.get("add", []) or [],
                delete=a.get("del", a.get("delete", [])) or [],
                nl=a.get("nl", a["name"]),
                num_pre=a.get("num_pre", []) or [],
                num_eff=a.get("num_eff", []) or [],
                cond=cond_blocks or [],
                duration=a.get("duration", None),
            )
        return DomainSpec(y["domain"], types, preds, actions, fls)
