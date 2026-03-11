from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
from sere.pddl.domain_spec import DomainSpec, Predicate
from sere.core.world_state import WorldState
from sere.core.semantics import eval_clause
from sere.core.pddl_env.run_mode import RunMode
import fnmatch
import re
from enum import Enum

class VisibilityScope(Enum):
    ALL = "all"
    ROOM = "room"

@dataclass
class PromptFormatterConfig:
    # Visibility
    visibility: VisibilityScope = VisibilityScope.ALL
    show_affordances: bool = True
    show_footer: bool = False

    # Misc presentation
    show_briefing: bool = True
    show_objects_in_sysprompt: bool = True
    show_domain_pddl: bool = False  # include raw domain.pddl in system prompt
    observer_robot: Optional[str] = None
    multi_agent: bool = False

    # Goal / fluents / messages
    show_goal: bool = True
    show_fluents: bool = True
    show_messages: bool = True
    messages_inline: bool = True

    # Fluents formatting
    fluents_precision: int = 2
    show_fluent_deltas: bool = True
    visible_fluents: Optional[List[str]] = None  # glob patterns (None -> ["*"])


class PromptFormatter:
    def __init__(self, domain: DomainSpec, config: Optional[PromptFormatterConfig] = None):
        self.cfg = config or PromptFormatterConfig()
        if self.cfg.visible_fluents is None:
            self.cfg.visible_fluents = ["*"]
        self.domain = domain


    # ---------- Small helpers ----------
    def _pddl(self, pred):  # pred = (name, args)
        n, a = pred
        return f"({n}{'' if not a else ' ' + ' '.join(a)})"

    def _is_numeric_clause(self, s: str) -> bool:
        s = s.strip()
        if s.startswith("(not"):
            inner = s[4:].strip()
            if inner.startswith("(") and inner.endswith(")"):
                s = inner

        m = re.match(r"^\(\s*([^\s()]+)", s)
        if not m:
            return False
        op = m.group(1)
        return op in {">", "<", ">=", "<=", "=", "+", "-", "*", "/"}

    def _format_objects_name_type(self, world: WorldState) -> str:
        lines = []
        for sym, types in sorted(world.objects.items()):
            type_str = ", ".join(sorted(types)) if types else "untyped"
            lines.append(f"{sym}: {{{type_str}}}")
        return "\n".join(lines) if lines else "(none)"

    def _scoped_view(
        self,
        world: WorldState,
        static_facts: Optional[Set[Predicate]] = None,
    ) -> tuple[WorldState, Set[Predicate], Optional[str]]:
        """
        Return a *scoped* view of (world, static_facts) based on visibility,
        plus the current room token (if applicable).
        """
        static_facts = static_facts or set()

        match self.cfg.visibility:
            case VisibilityScope.ALL:
                return world, static_facts, None

            case VisibilityScope.ROOM:
                room = self._robot_room(world)
                if room is None:
                    return world, static_facts, None

                kept_facts = {fa for fa in world.facts if self._fact_visible_in_room(world, fa, room)}
                kept_static = {sf for sf in static_facts if self._fact_visible_in_room(world, sf, room)}
                kept_fluents = {
                    (fname, args): v
                    for (fname, args), v in world.fluents.items()
                    if self._fluent_visible_in_room(world, fname, args, room)
                }

                visible_syms: set[str] = {room}
                visible_syms |= set(self._robots(world))
                cache: Dict[str, Optional[str]] = {}
                for sym in world.objects.keys():
                    if self._loc_of(world, sym, cache) == room:
                        visible_syms.add(sym)
                for (_p, args) in kept_facts:
                    for tok in args:
                        if tok in world.objects:
                            visible_syms.add(tok)
                for (_p, args) in kept_static:
                    for tok in args:
                        if tok in world.objects:
                            visible_syms.add(tok)
                for (_f, args) in kept_fluents.keys():
                    for tok in args:
                        if tok in world.objects:
                            visible_syms.add(tok)

                sliced_objects: Dict[str, set] = {}
                for sym in visible_syms:
                    if sym in world.objects:
                        tys = world.objects[sym]
                        sliced_objects[sym] = set(tys) if isinstance(tys, (set, list, tuple)) else {str(tys)}

                scoped_world = WorldState(
                    domain=world.domain,
                    objects=sliced_objects,
                    facts=kept_facts,
                    fluents=kept_fluents,
                )
                return scoped_world, kept_static, room


            case _:
                raise NotImplementedError(f"Unhandled visibility scope: {self.cfg.visibility}")


    def _ensure_object_registry(self, world: WorldState, static_facts: Set[Predicate]) -> None:
        """
        Ensure world.objects has type memberships inferred from current facts,
        using the domain's predicate signatures. Idempotent and non-destructive.
        """
        if not self.domain or not getattr(self.domain, "predicates", None):
            return

        pred_arg_types: Dict[str, List[str]] = {
            name: [typ for (_var, typ) in spec.args]
            for name, spec in self.domain.predicates.items()
        }

        def _add(sym: str, typ: str) -> None:
            if not typ:
                return
            s = world.objects.setdefault(sym, set())
            if isinstance(s, set):
                s.add(typ)
                for sup in self.domain.supertypes(typ):
                    s.add(sup)
            else:
                world.objects[sym] = {str(s), typ}
                for sup in self.domain.supertypes(typ):
                    world.objects[sym].add(sup)

        for (pname, args) in list(world.facts) + list(static_facts or set()):
            typs = pred_arg_types.get(pname)
            if not typs:
                continue
            for tok, typ in zip(args, typs):
                _add(tok, typ)


    # ---- Room helpers ----
    def _robots(self, world: WorldState) -> list[str]:
        return [s for s, tys in world.objects.items() if any(str(t).lower() == "robot" for t in (tys or []))]

    def _robot_room(self, world: WorldState) -> Optional[str]:
        rob = self.cfg.observer_robot or (self._robots(world)[0] if self._robots(world) else None)
        if not rob:
            return None
        locs = [a[1] for (p, a) in world.facts if p == "at" and len(a) == 2 and a[0] == rob]
        return locs[0] if len(locs) == 1 else None

    def _loc_of(self, world: WorldState, sym: str, cache: Optional[Dict[str, Optional[str]]] = None) -> Optional[str]:
        c = cache if cache is not None else {}
        if sym in c:
            return c[sym]
        locs = world.locations_of(sym)
        loc = next(iter(locs)) if len(locs) == 1 else None
        c[sym] = loc
        return loc

    def _fact_visible_in_room(self, world: WorldState, fact: Predicate, room: Optional[str]) -> bool:
        if room is None:
            return True
        name, args = fact
        if name == "adjacent":
            return False
        if name == "at" and len(args) == 2:
            return True
        if name == "obj-at" and len(args) == 2:
            return args[1] == room
        obj_args = [a for a in args if a in world.objects]
        cache: Dict[str, Optional[str]] = {}
        return any(self._loc_of(world, s, cache) == room for s in obj_args)

    def _fluent_visible_in_room(self, world: WorldState, fname: str, args: Tuple[str, ...], room: Optional[str]) -> bool:
        if room is None:
            return True
        if fname in {"energy", "battery-cap"} and args and args[0] in self._robots(world):
            return True
        if args and args[0] in world.objects:
            return self._loc_of(world, args[0], {}) == room
        return False

    # ---------- System prompt ----------
    def build_system_prompt(
        self,
        *,
        world: WorldState,
        static_facts: Optional[Set[Predicate]] = None,
        time_limit: float | None = None,
        run_mode: RunMode
    ) -> str:

        if not self.cfg.show_briefing:
            return ""

        mode: RunMode = run_mode

        scoped_world, scoped_static, _ = self._scoped_view(world, static_facts or set())

        has_energy = any(name == "energy" for (name, _args) in scoped_world.fluents.keys())
        has_time = any((a.duration is not None) or (a.duration_var is not None)
                    for a in self.domain.actions.values())

        parts: list[str] = []

        # High-level briefing + mode banner
        expl: list[str] = []
        expl.append("You are controlling a robot in a simulated world.")
        if has_energy:
            expl.append(
                "The robot has limited energy. The current energy is shown in the header as "
                "Energy: r1 current/capacity. When energy is zero the robot cannot perform most actions. "
                "Use the recharge action at a charging location to restore energy up to the battery capacity. "
                "Energy never exceeds the capacity. Plan actions so the robot does not run out of energy."
            )
        if has_time:
            expl.append(
                "Each action takes time. The header shows Time. " +
                (f"This task has a strict time limit of {time_limit:.2f}. Exceeding it fails the task. "
                if time_limit is not None else
                "Some tasks have a time limit. Exceeding it fails the task. ") +
                "Choose efficient actions."
            )
        expl.append(
            "The header always shows the current step number and the maximum allowed steps. "
            "If time is enabled it also shows total elapsed time. If energy is modeled it shows the energy level "
            "for each robot, and the battery capacity if defined."
        )

        parts.append(f"You are in {mode.value.replace('_','-').upper()} mode.")
        if self.cfg.multi_agent:
            expl.append(
                "Multi-agent mode: submit one action per robot each step. "
                "Use (idle r) if a robot should do nothing."
            )

        match mode:
            case RunMode.OPEN_LOOP:
                expl.append(
                    "Submit a complete plan as one block: "
                    "(a1 ...)(a2 ...).... The environment executes sequentially, "
                    "then the episode ends."
                )
                reply_hint = "Reply with a full plan: (a1 ...)(a2 ...)..., e.g. (pick-up r1 leaf) (move r1 A B) (steep-tea r1 leaf mug)."
            case RunMode.BATCH:
                expl.append(
                    "You may submit multiple actions at once as "
                    "(a1 ...)(a2 ...).... The environment executes them in order and returns."
                )
                reply_hint = "Reply with one or more actions: (a1 ...)(a2 ...)..., e.g. (move r1 A B) (pour r1 kettle mug)."
            case RunMode.INTERACTIVE:
                expl.append(
                    "Submit exactly one action per step as "
                    "(action arg1 arg2 ...)."
                )
                reply_hint = "Reply with exactly one action: (action arg1 arg2 ...), e.g. (move r1 A B)."
            case _:
                raise NotImplementedError(f"Unhandled run mode: {mode}")

        if self.cfg.multi_agent:
            reply_hint += " One action per robot; use (idle r) for no-op."

        parts.append(" ".join(expl))

        # -------- Domain specification --------
        parts.append(f"You are controlling the robot in the '{self.domain.name}' domain.")
        parts.append(reply_hint)

        if self.cfg.show_domain_pddl and getattr(self.domain, "pddl_source", ""):
            parts += [
                "",
                "Domain definition (PDDL):",
                self.domain.pddl_source.strip(),
            ]
        else:
            actions = self._format_action_catalog()
            parts += [
                "",
                "Actions (signature [costs]):",
                actions or "(none)",
            ]

        # -------- Objects & Statics (respect visibility) --------
        scoped_world, scoped_static, _ = self._scoped_view(world, static_facts or set())

        if self.cfg.show_objects_in_sysprompt:
            parts += ["", "Objects (name - type):", self._format_objects_name_type(scoped_world)]

        lines: list[str] = []
        for pred in sorted(scoped_static):
            lines.append(f"- {self._pddl(pred)}")
        parts += ["", "Statics (do not change):", "\n".join(lines) if lines else "(none)"]

        return "\n".join(parts)



    # ---------- Observation ----------
    def format_obs(
        self,
        *,
        world: WorldState,
        steps: int,
        max_steps: int,
        time_val: float,
        durations_on: bool,
        messages: List[str],
        affordances: List[str],
        time_limit: float | None,
        termination_rules: Optional[List[dict]] = None,
        run_mode: RunMode
        ) -> str:
        mode: RunMode = run_mode

        scoped_world, _scoped_static, _room = self._scoped_view(world, static_facts=None)

        # ----- State -----
        raw_facts = [fa for fa in sorted(scoped_world.facts) if fa[0] != "adjacent"]
        state_lines = [f"- {self._pddl(pred)}" for pred in raw_facts]
        state_txt = "State:\n" + ("\n".join(state_lines) if state_lines else "(none)")

        # ----- Goal -----
        goal_txt = ""
        success_line = self._render_success_goal_line(termination_rules)
        if success_line:
            goal_txt = "Goal:\n" + success_line

        # ----- Fluents -----
        fl_txt = ""
        if self.cfg.show_fluents and scoped_world.fluents:
            prec = max(0, int(self.cfg.fluents_precision))
            rows = []
            for (name, args), val in sorted(scoped_world.fluents.items()):
                if not any(fnmatch.fnmatch(name, pat) for pat in (self.cfg.visible_fluents or ["*"])):
                    continue
                argstr = "" if not args else " " + " ".join(args)
                rows.append(f"- (= ({name}{argstr}) {val:.{prec}f})")
            fl_txt = "Fluents:\n" + ("\n".join(rows) if rows else "")

        # ----- Affordances -----
        aff_txt = ""
        if self.cfg.show_affordances and affordances:
            lines = [f"- {a}" for a in affordances]
            aff_txt = "Valid moves:\n" + "\n".join(lines)

        # ----- Messages -----
        msg_txt = ""
        if self.cfg.show_messages and messages:
            msg_txt = "\n".join(messages) if self.cfg.messages_inline else ("Messages:\n  - " + "\n  - ".join(messages))

        # ----- Header -----
        limit = "" if time_limit is None else f"/{time_limit:.2f}"
        time_txt = f" | Time: {time_val:.2f}{limit}" if durations_on else ""
        energy_bits = []
        for sym, types in sorted(scoped_world.objects.items()):
            if any(t.lower() == "robot" for t in (types or [])):
                e = scoped_world.get_fluent("energy", (sym,))
                cap = scoped_world.get_fluent("battery-cap", (sym,))
                has_cap = (("battery-cap", (sym,)) in scoped_world.fluents)
                energy_bits.append(f"{sym} {e:.2f}/{cap:.2f}" if has_cap else f"{sym} {e:.2f}")
        energy_txt = (" | Energy: " + ", ".join(energy_bits)) if energy_bits else ""
        header = f"Steps: {steps}/{max_steps}{time_txt}{energy_txt}"

        # ----- Footer (mode-specific) -----
        tail = ""
        match mode:
            case RunMode.OPEN_LOOP:
                tail = "Submit a full plan as (a1 ...)(a2 ...).... Episode will end after execution."
            case RunMode.BATCH:
                tail = "You may submit multiple actions: (a1 ...)(a2 ...)...."
            case RunMode.INTERACTIVE:
                tail = "Reply with (action args)."
            case _:
                raise NotImplementedError(f"Unhandled run mode: {mode}")

        if self.cfg.multi_agent:
            tail = tail.rstrip(".") + " One action per robot; use (idle r) for no-op."

        footer_block = tail if self.cfg.show_footer else ""
        prompt_line = "" if self.cfg.show_footer else "What will be your next move?"

        return "\n\n".join(
            p for p in [
                header,
                msg_txt,
                state_txt,
                goal_txt,
                fl_txt,
                aff_txt,
                footer_block,
                prompt_line,
            ] if p
        ).strip()



    # ---------- Affordances ----------
    def generate_affordances(
        self,
        world: WorldState,
        static_facts: Set[Predicate],
        enable_numeric: bool = True,
    ) -> List[str]:
        if not self.cfg.show_affordances:
            return []

        scoped_world, scoped_static, _room = self._scoped_view(world, static_facts)
        self._ensure_object_registry(scoped_world, scoped_static)

        def _is_type(sym: str, typ: str) -> bool:
            t = scoped_world.objects.get(sym)
            if isinstance(t, str):
                return self.domain.is_subtype(t, typ)
            if isinstance(t, (set, list, tuple)):
                return any(self.domain.is_subtype(str(tt), typ) for tt in t)
            return False

        by_type: Dict[str, List[str]] = {}
        for sym in scoped_world.objects.keys():
            t = scoped_world.objects[sym]
            if isinstance(t, str):
                by_type.setdefault(t, []).append(sym)
            elif isinstance(t, (set, list, tuple)):
                for tt in t:
                    by_type.setdefault(str(tt), []).append(sym)

        def cartesian(lists: List[List[str]]) -> List[List[str]]:
            res = [[]]
            for lst in lists:
                if not lst:
                    return []
                res = [r + [x] for r in res for x in lst]
            return res

        afford: List[str] = []
        for act in sorted(self.domain.actions.values(), key=lambda x: x.name):
            num_positions = [i for i, (_v, typ) in enumerate(act.params) if str(typ).lower() == "number"]

            pools: List[List[str]] = []
            for i, (_var, typ) in enumerate(act.params):
                if str(typ).lower() == "number":
                    pools.append(["<n>"])
                else:
                    pool = by_type.get(typ) or [s for s in scoped_world.objects if _is_type(s, typ)]
                    pools.append(pool)

            combos = cartesian(pools)
            if not combos:
                continue

            derived_cache: Dict[Tuple[str, Tuple[str, ...]], bool] = {}

            for args in combos:
                bind = {var: val for (var, _), val in zip(act.params, args)}

                has_number = len(num_positions) > 0
                numeric_ok = enable_numeric and (not has_number)

                ok = True
                for pre in (act.pre or []):
                    if has_number and self._is_numeric_clause(pre):
                        continue
                    if not eval_clause(
                        scoped_world,
                        scoped_static,
                        pre,
                        bind,
                        enable_numeric=numeric_ok,
                        derived_cache=derived_cache,
                    ):
                        ok = False
                        break
                if not ok:
                    continue

                shown_args: List[str] = []
                for i, val in enumerate(args):
                    shown_args.append("<n>" if i in num_positions else val)

                if shown_args:
                    afford.append(f"({act.name} {' '.join(shown_args)})")
                else:
                    afford.append(f"({act.name})")

        if self.cfg.multi_agent:
            for sym, types in sorted(scoped_world.objects.items()):
                if any(t.lower() == "robot" for t in (types or [])):
                    afford.append(f"(idle {sym})")

        return afford



    # ---------- Helpers ----------
    def _format_action_catalog(self) -> str:
        rows = []
        for a in sorted(self.domain.actions.values(), key=lambda x: x.name):
            sig = f"{a.name}(" + ", ".join(f"{v}:{t}" for v, t in a.params) + ")"
            energy = self._energy_hint(a)
            dur = self._duration_hint(a)
            rows.append(f"- {sig}  [{energy}; {dur}]")
        if self.cfg.multi_agent:
            rows.append("- idle(r:robot)  [0; 0]")
        return "\n".join(rows)


    # ---------- Success goal rendering from termination_rules ----------
    def _render_success_goal_line(self, termination_rules: Optional[List[dict]]) -> str:
        if not self.cfg.show_goal or not termination_rules:
            return ""

        success = None
        for r in termination_rules:
            if str(r.get("outcome", "")).lower() == "success":
                success = r
                break
        if not success:
            return ""

        when = success.get("when")
        if when is None:
            return ""

        mode = "single"
        exprs: List[str] = []
        if isinstance(when, str):
            exprs = [when]
        elif isinstance(when, dict):
            if "any" in when and isinstance(when["any"], list):
                mode = "any"
                exprs = [str(x) for x in when["any"]]
            elif "all" in when and isinstance(when["all"], list):
                mode = "all"
                exprs = [str(x) for x in when["all"]]
        if not exprs:
            return ""

        joiner = " or " if mode == "any" else (" and " if mode == "all" else "")
        return f"- {joiner.join(exprs)}"

    # ---------- Cost/introspection helpers ----------
    _NUM_EFF_RX = re.compile(
        r"^\(\s*(increase|decrease|assign)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+([^\s()]+)\s*\)$"
    )

    def _energy_hint(self, act) -> str:
        def _outcome_status(oc) -> str:
            if oc is None:
                return ""
            if isinstance(oc, dict):
                return str(oc.get("status", "")).lower()
            return str(getattr(oc, "status", "")).lower()

        def _outcome_num_eff(oc):
            if oc is None: return []
            if isinstance(oc, dict): return oc.get("num_eff") or []
            return getattr(oc, "num_eff", None) or []

        def _action_num_eff(a):
            return getattr(a, "num_eff", None) or []

        def render_delta(num_eff_list) -> str:
            total = 0.0
            symbolic = []

            for expr in (num_eff_list or []):
                s = str(expr).strip()
                m = self._NUM_EFF_RX.match(s)
                if not m: continue
                op, fname, _argstr, rhs = m.groups()
                if fname != "energy": continue
                if op == "assign":
                    return f"energy \u2192 {rhs.strip()}"
                rhs = rhs.strip()
                try:
                    val = float(rhs)
                    if op == "increase": total += val
                    elif op == "decrease": total -= val
                except Exception:
                    symbolic.append((op, rhs))
            if symbolic and abs(total) < 1e-12:
                op, rhs = symbolic[-1]
                if op == "increase": return f"energy +{rhs}"
                if op == "decrease": return f"energy -{rhs}"
                if op == "assign":   return f"energy \u2192 {rhs}"
            sign = "+" if total >= 0 else ""
            return f"energy {sign}{total:.3f}"

        outcomes = list(getattr(act, "outcomes", []) or [])

        chosen = None
        for oc in outcomes:
            if _outcome_status(oc) == "success":
                chosen = oc
                break

        if chosen is None and outcomes:
            chosen = outcomes[0]

        num_eff = _outcome_num_eff(chosen)
        if not num_eff:
            num_eff = _action_num_eff(act)

        return render_delta(num_eff)



    def _duration_hint(self, act) -> str:
        vname = getattr(act, "duration_var", None)
        unit  = getattr(act, "duration_unit", None)
        dur   = getattr(act, "duration", None)
        if isinstance(vname, str) and isinstance(unit, (int, float)):
            u = f"{unit:.3f}".rstrip("0").rstrip(".")
            return f"duration {u}×{vname}"
        if isinstance(dur, (int, float)):
            d = f"{float(dur):.3f}".rstrip("0").rstrip(".")
            return f"duration {d}"
        return "duration default"
