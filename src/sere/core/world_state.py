from dataclasses import dataclass, field
from typing import Dict, Set, Optional, AbstractSet, List, Tuple
from ..pddl.domain_spec import DomainSpec, Predicate

def lit(p: str, *args: str) -> Predicate:
    return (p, tuple(args))

@dataclass
class WorldState:
    domain: DomainSpec
    # symbol -> set of types (multi-typing)
    objects: Dict[str, Set[str]] = field(default_factory=dict)
    facts: Set[Predicate] = field(default_factory=set)
    fluents: Dict[Tuple[str, Tuple[str, ...]], float] = field(default_factory=dict)

    def add_object(self, sym: str, typ: str):
        s = self.objects.setdefault(sym, set())
        s.add(typ)

    def holds(self, p: Predicate) -> bool:
        name, args = p
        if name == "co-located":
            x, y = args
            lx = {a[1] for (pred, a) in self.facts if pred == "at" and len(a) == 2 and a[0] == x} \
            | {a[1] for (pred, a) in self.facts if pred == "obj-at" and len(a) == 2 and a[0] == x}
            ly = {a[1] for (pred, a) in self.facts if pred == "at" and len(a) == 2 and a[0] == y} \
            | {a[1] for (pred, a) in self.facts if pred == "obj-at" and len(a) == 2 and a[0] == y}
            return bool(lx & ly)
        return p in self.facts


    def get_fluent(self, name: str, args: Tuple[str, ...]) -> float:
        return self.fluents.get((name, args), 0.0)

    def set_fluent(self, name: str, args: Tuple[str, ...], value: float):
        self.fluents[(name, args)] = float(value)

    def apply(self, add: List[Predicate], delete: List[Predicate]):
        for d in delete: self.facts.discard(d)
        for a in add: self.facts.add(a)

    def check_preconds(self, pre: List[Predicate], extra: Optional[AbstractSet[Predicate]] = None) -> List[Predicate]:
        facts = self.facts | (extra or set())
        missing: List[Predicate] = []
        for p in pre:
            if p in facts:
                continue
            # consult derived semantics (e.g., co-located) if not a literal fact
            if self.holds(p):
                continue
            missing.append(p)
        return missing


    def validate_invariants(self) -> List[str]:
        errs = []
        loc_map, in_map = {}, {}
        for (pred, args) in self.facts:
            if pred == "obj-at":
                loc_map.setdefault(args[0], set()).add(args[1])
            if pred == "in":
                in_map.setdefault(args[0], set()).add(args[1])

        for o, locs in loc_map.items():
            if len(locs) > 1:
                errs.append(f"{o}: multiple 'obj-at' locations {sorted(locs)}")

        for o in set(loc_map) & set(in_map):
            errs.append(f"{o}: both 'obj-at' and 'in' present")

        return errs

    def to_problem_pddl(self, name: str, static_facts: Set[Predicate], goals: List[Predicate]) -> str:
        init = self.facts | static_facts
        init_str = " ".join(f"({p} {' '.join(a)})" for (p,a) in sorted(init))
        goals_str = " ".join(f"({g[0]} {' '.join(g[1])})" for g in goals)
        objs = " ".join(sorted(self.objects))
        return f"(define (problem {name})\n  (:domain {self.domain.name})\n  (:objects {objs})\n  (:init {init_str})\n  (:goal (and {goals_str}))\n)"
