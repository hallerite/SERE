from typing import Protocol, List, Set, Tuple, Dict
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

class AssemblyInvariants:
    def validate(self, w: WorldState, statics: Set[Predicate]) -> List[str]:
        errs: List[str] = []

        # Collect state
        equipped_by_r: Dict[str, Set[str]] = {}
        equipped_tool_holders: Dict[str, Set[str]] = {}
        in_map: Dict[str, Set[str]] = {}
        at_map: Dict[str, Set[str]] = {}
        holding_map: Dict[str, Set[str]] = {}
        installed_map: Dict[str, Set[str]] = {}
        fastened_map: Dict[str, Set[str]] = {}

        for (p, a) in w.facts:
            if p == "equipped":
                r, t = a
                equipped_by_r.setdefault(r, set()).add(t)
                equipped_tool_holders.setdefault(t, set()).add(r)
            elif p == "in":
                in_map.setdefault(a[0], set()).add(a[1])        # obj -> {containers}
            elif p == "obj-at":
                at_map.setdefault(a[0], set()).add(a[1])        # obj -> {locations}
            elif p == "holding":
                holding_map.setdefault(a[1], set()).add(a[0])   # obj -> {robots}
            elif p == "installed":
                installed_map.setdefault(a[0], set()).add(a[1]) # part -> {assemblies}
            elif p == "fastened":
                fastened_map.setdefault(a[0], set()).add(a[1])  # part -> {assemblies}

        # A) Tool equip sanity
        for r, tools in equipped_by_r.items():
            if len(tools) > 1:
                errs.append(f"{r}: equipped with multiple tools {sorted(tools)}")
        for t, robots in equipped_tool_holders.items():
            if len(robots) > 1:
                errs.append(f"{t}: equipped by multiple robots {sorted(robots)}")

        # Equipped tool cannot simultaneously be held / stored / placed
        for t in equipped_tool_holders:
            if t in holding_map:
                errs.append(f"{t}: equipped and held simultaneously")
            if t in in_map:
                errs.append(f"{t}: equipped but also in containers {sorted(in_map[t])}")
            if t in at_map:
                errs.append(f"{t}: equipped but also has obj-at {sorted(at_map[t])}")

        # B) Part attachment sanity
        # Part cannot be installed/fastened to multiple assemblies at once
        for p, asms in installed_map.items():
            if len(asms) > 1:
                errs.append(f"{p}: installed on multiple assemblies {sorted(asms)}")
        damaged_parts = {a[0] for (pred, a) in w.facts if pred == "damaged"}
        for p, asms in fastened_map.items():
            if p in damaged_parts:
                continue  # allowed: fastened but not installed when damaged
            missing = asms - installed_map.get(p, set())
            if missing:
                errs.append(f"{p}: fastened to {sorted(missing)} but not installed")

        # Held part cannot be installed/fastened
        for p in holding_map:
            if p in installed_map:
                errs.append(f"{p}: held while installed on {sorted(installed_map[p])}")
            if p in fastened_map:
                errs.append(f"{p}: held while fastened to {sorted(fastened_map[p])}")

        # C) QC flags – make them mutually exclusive
        defective = {a[0] for (pred, a) in w.facts if pred == "defective"}
        rework    = {a[0] for (pred, a) in w.facts if pred == "needs-rework"}
        for asm in sorted(defective & rework):
            errs.append(f"{asm}: both 'defective' and 'needs-rework' set")

        return errs
