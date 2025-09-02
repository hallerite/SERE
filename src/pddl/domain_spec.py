from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml

Predicate = Tuple[str, Tuple[str, ...]]

@dataclass
class PredicateSpec:
    name: str
    args: List[Tuple[str, str]]  # (var, type)
    nl: str
    static: bool = False

@dataclass
class ActionSpec:
    name: str
    params: List[Tuple[str, str]]  # (var, type)
    pre: List[str]                  # e.g., ["(at ?r ?l)"]
    add: List[str]
    delete: List[str]
    nl: str

@dataclass
class DomainSpec:
    name: str
    types: Dict[str, str]                     # subtype -> supertype ("" if root)
    predicates: Dict[str, PredicateSpec]
    actions: Dict[str, ActionSpec]

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
        for p in y["predicates"]:
            args = [(a["name"], a["type"]) for a in p.get("args", [])]
            preds[p["name"]] = PredicateSpec(
                name=p["name"],
                args=args,
                nl=p.get("nl", p["name"]),
                static=p.get("static", False)
            )
        # actions
        actions = {}
        for a in y["actions"]:
            params = []
            for d in a.get("params", []):
                ((var, typ),) = d.items()
                params.append((var, typ))
            actions[a["name"]] = ActionSpec(
                name=a["name"],
                params=params,
                pre=a.get("pre", []),
                add=a.get("add", []),
                delete=a.get("del", a.get("delete", [])),
                nl=a.get("nl", a["name"])
            )
        return DomainSpec(y["domain"], types, preds, actions)
