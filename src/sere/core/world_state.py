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
        errs: List[str] = []
        loc_map: Dict[str, Set[str]] = {}
        in_map: Dict[str, Set[str]] = {}
        holding_map: Dict[str, Set[str]] = {}
        robot_loc: Dict[str, Set[str]] = {}

        for (pred, args) in self.facts:
            if pred == "obj-at":
                loc_map.setdefault(args[0], set()).add(args[1])
            elif pred == "in":
                in_map.setdefault(args[0], set()).add(args[1])
            elif pred == "holding":
                # args = (r, o) → index 1 is the object
                holding_map.setdefault(args[1], set()).add(args[0])
            elif pred == "at" and len(args) == 2:
                robot_loc.setdefault(args[0], set()).add(args[1])

        # object cannot be at multiple locations
        for o, locs in loc_map.items():
            if len(locs) > 1:
                errs.append(f"{o}: multiple 'obj-at' locations {sorted(locs)}")

        # object cannot be both at a location and inside a container
        for o in set(loc_map) & set(in_map):
            errs.append(f"{o}: both 'obj-at' and 'in' present")

        # held object cannot also be 'in' or 'obj-at'
        for o in holding_map:
            if o in loc_map:
                errs.append(f"{o}: held and also 'obj-at' present")
            if o in in_map:
                errs.append(f"{o}: held and also 'in' present")

        # object cannot be inside multiple containers
        for o, cs in in_map.items():
            if len(cs) > 1:
                errs.append(f"{o}: in multiple containers {sorted(cs)}")

        # robot must be at exactly one location
        for r, locs in robot_loc.items():
            if len(locs) != 1:
                errs.append(f"{r}: robot has {len(locs)} locations {sorted(locs)}")

        return errs


    def to_problem_pddl(self, name: str, static_facts: Set[Predicate], goals: List[Predicate]) -> str:
        init = self.facts | static_facts
        init_str = " ".join(f"({p} {' '.join(a)})" for (p,a) in sorted(init))
        goals_str = " ".join(f"({g[0]} {' '.join(g[1])})" for g in goals)
        objs = " ".join(sorted(self.objects))
        return f"(define (problem {name})\n  (:domain {self.domain.name})\n  (:objects {objs})\n  (:init {init_str})\n  (:goal (and {goals_str}))\n)"
