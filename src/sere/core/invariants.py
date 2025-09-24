from typing import Protocol, List, Set, Tuple
from .world_state import WorldState

Predicate = Tuple[str, Tuple[str, ...]]

class InvariantPlugin(Protocol):
    def validate(self, world: WorldState, static_facts: Set[Predicate]) -> List[str]: ...

class KitchenInvariants:
    def validate(self, w: WorldState, statics: Set[Predicate]) -> List[str]:
        errs: List[str] = []

        def S(pred: str, *args: str) -> Predicate:
            return (pred, tuple(args))

        def has_static(pred: str, *args: str) -> bool:
            return S(pred, *args) in statics

        # open ⇒ openable (authoring/logic guard)
        for (p, a) in w.facts:
            if p == "open":
                c = a[0]
                if not has_static("openable", c):
                    errs.append(f"{c}: 'open' but no static 'openable'.")

        # needs-open ⇒ openable (authoring guard)
        for (p, a) in statics:
            if p == "needs-open":
                c = a[0]
                if not has_static("openable", c):
                    errs.append(f"{c}: static 'needs-open' but missing static 'openable'.")

        # has-hot-water/tea-ready ⇒ temp ≥ 80 and not spilled
        hot = [a[0] for (p, a) in w.facts if p == "has-hot-water"]
        ready = [a[0] for (p, a) in w.facts if p == "tea-ready"]
        for m in set(hot + ready):
            temp = w.get_fluent("water-temp", (m,))
            if temp < 80.0:
                errs.append(f"{m}: marked hot/ready but water-temp={temp:.1f} < 80.")
            if S("spilled", m) in w.facts:
                errs.append(f"{m}: cannot be 'spilled' while hot/ready.")

        # adjacency symmetry in statics
        for (p, a) in statics:
            if p == "adjacent":
                l1, l2 = a
                if not has_static("adjacent", l2, l1):
                    errs.append(f"adjacent not symmetric: have ({l1},{l2}) but missing ({l2},{l1}).")

        return errs
