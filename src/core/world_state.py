from dataclasses import dataclass, field
from typing import Dict, Set, Optional, AbstractSet, List
from ..pddl.domain_spec import DomainSpec, Predicate

def lit(p: str, *args: str) -> Predicate:
    return (p, tuple(args))

@dataclass
class WorldState:
    domain: DomainSpec
    objects: Dict[str, str] = field(default_factory=dict)   # symbol -> type
    facts: Set[Predicate] = field(default_factory=set)

    def add_object(self, sym: str, typ: str):
        self.objects[sym] = typ

    def holds(self, p: Predicate) -> bool:
        return p in self.facts

    def apply(self, add: List[Predicate], delete: List[Predicate]):
        for d in delete: self.facts.discard(d)
        for a in add: self.facts.add(a)

    def check_preconds(self, pre: List[Predicate], extra: Optional[AbstractSet[Predicate]] = None) -> List[Predicate]:
        facts = self.facts | (extra or set())
        return [p for p in pre if p not in facts]

    def validate_invariants(self) -> List[str]:
        errs = []
        # Example: object cannot be both at a location and inside a container
        loc_map, in_map = {}, {}
        for (pred,args) in self.facts:
            if pred == "obj-at": loc_map.setdefault(args[0], set()).add(args[1])
            if pred == "in":     in_map.setdefault(args[0], set()).add(args[1])
        for o in set(loc_map) & set(in_map):
            errs.append(f"{o}: both 'obj-at' and 'in' present")
        return errs

    def to_problem_pddl(self, name: str, static_facts: Set[Predicate], goals: List[Predicate]) -> str:
        init = self.facts | static_facts
        init_str = " ".join(f"({p} {' '.join(a)})" for (p,a) in sorted(init))
        goals_str = " ".join(f"({g[0]} {' '.join(g[1])})" for g in goals)
        objs = " ".join(sorted(self.objects))
        return f"(define (problem {name})\n  (:domain {self.domain.name})\n  (:objects {objs})\n  (:init {init_str})\n  (:goal (and {goals_str}))\n)"
