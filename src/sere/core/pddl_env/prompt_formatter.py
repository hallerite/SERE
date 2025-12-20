from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, Callable
from sere.pddl.domain_spec import DomainSpec, Predicate
from sere.pddl.nl_mapper import NLMapper
from sere.core.world_state import WorldState
from sere.core.semantics import eval_clause
from sere.core.pddl_env.run_mode import RunMode
import fnmatch
import re
import random
from enum import Enum

class VisibilityScope(Enum):
    ALL = "all"
    ROOM = "room"

@dataclass
class PromptFormatterConfig:
    # Unified rendering: system prompt + observations
    display_nl: bool = True   # True → NL+PDDL; False → PDDL-only
    # NL variation controls
    nl_stochastic: bool = False               # False → always first NL template; True → sample a variant
    nl_rng_seed: Optional[int] = None

    # Visibility
    visibility: VisibilityScope = VisibilityScope.ALL   # ALL or ROOM
    show_affordances: bool = True                      # render affordances in obs
    show_footer: bool = False

    # Misc presentation
    show_briefing: bool = True
    show_objects_in_sysprompt: bool = True
    observer_robot: Optional[str] = None

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
        rng = random.Random(self.cfg.nl_rng_seed) if self.cfg.nl_rng_seed is not None else None
        self.nl = NLMapper(domain, stochastic=self.cfg.nl_stochastic, rng=rng)


    # ---------- Small helpers ----------
    def _pddl(self, pred):  # pred = (name, args)
        n, a = pred
        return f"({n}{'' if not a else ' ' + ' '.join(a)})"

    def _nl_pred(self, pred):
        try:
            return self.nl.pred_to_text(pred)
        except Exception:
            return None

    def _inline(self, pddl_str: str, nl_str: str | None) -> str:
        return f"- {pddl_str}" if not (self.cfg.display_nl and nl_str) else f"- {pddl_str} – {nl_str}"

    def _is_numeric_clause(self, s: str) -> bool:
        """
        Heuristic: treat comparisons/arith forms as numeric preconditions.
        Used to skip numeric guards when an action has a 'number' param and we
        template that param as '<n>'.
        """
        s = s.strip()
        # unwrap a simple (not (...)) so we inspect the inner operator
        if s.startswith("(not"):
            inner = s[4:].strip()
            if inner.startswith("(") and inner.endswith(")"):
                s = inner[1:-1].strip()

        m = re.match(r"^\(\s*([^\s()]+)", s)
        if not m:
            return False
        op = m.group(1)
        # common numeric operators in preconditions
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

        - VisibilityScope.ALL : pass-through
        - VisibilityScope.ROOM: filter to facts/fluents 'visible in room'
        """
        static_facts = static_facts or set()

        match self.cfg.visibility:
            case VisibilityScope.ALL:
                return world, static_facts, None

            case VisibilityScope.ROOM:
                room = self._robot_room(world)
                if room is None:
                    return world, static_facts, None

                # Filter dynamic facts / static facts / fluents
                kept_facts = {fa for fa in world.facts if self._fact_visible_in_room(world, fa, room)}
                kept_static = {sf for sf in static_facts if self._fact_visible_in_room(world, sf, room)}
                kept_fluents = {
                    (fname, args): v
                    for (fname, args), v in world.fluents.items()
                    if self._fluent_visible_in_room(world, fname, args, room)
                }

                # Compute visible symbols (robots, the room, and any object resolving to the room;
                # also anything referenced by kept facts/fluents/statics)
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

                # Slice the object registry (preserve type memberships)
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

        # predicate -> [arg_type0, arg_type1, ...]
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
            else:
                # tolerate odd entries (e.g., str) by normalizing to a set
                world.objects[sym] = {str(s), typ}

        # Walk dynamic + static facts and back-fill object types
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
        """Resolve an object's room using WorldState location resolution."""
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
            return True  # always show robot location
        if name == "obj-at" and len(args) == 2:
            return args[1] == room
        # generic rule: if any object arg resolves to this room, show it
        obj_args = [a for a in args if a in world.objects]
        cache: Dict[str, Optional[str]] = {}
        return any(self._loc_of(world, s, cache) == room for s in obj_args)

    def _fluent_visible_in_room(self, world: WorldState, fname: str, args: Tuple[str, ...], room: Optional[str]) -> bool:
        if room is None:
            return True
        # always show robot energy/battery
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

        mode: RunMode = run_mode  # already normalized by the env

        # Apply visibility scoping so objects/statics reflect the current view
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

        parts.append(" ".join(expl))

        # -------- Action catalog (always included) --------
        actions = self._format_action_catalog()
        parts += [
            f"You are controlling the robot in the '{self.domain.name}' domain.",
            reply_hint,
            "",
            "Actions (signature — description  [costs]):",
            actions or "(none)",
        ]

        # -------- Objects & Statics (respect visibility) --------
        scoped_world, scoped_static, _ = self._scoped_view(world, static_facts or set())

        if self.cfg.show_objects_in_sysprompt:
            parts += ["", "Objects (name - type):", self._format_objects_name_type(scoped_world)]

        lines: list[str] = []
        for pred in sorted(scoped_static):
            pddl = self._pddl(pred)
            nl = self._nl_pred(pred) if self.cfg.display_nl else None
            lines.append(self._inline(pddl, nl))
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
        mode: RunMode = run_mode  # guaranteed enum

        # Apply visibility scoping once
        scoped_world, _scoped_static, _room = self._scoped_view(world, static_facts=None)

        # ----- State (already scoped) -----
        raw_facts = [fa for fa in sorted(scoped_world.facts) if fa[0] != "adjacent"]
        state_lines: List[str] = []
        for pred in raw_facts:
            pddl = self._pddl(pred)
            nl = self._nl_pred(pred) if self.cfg.display_nl else None
            state_lines.append(self._inline(pddl, nl))
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
            lines = []
            for a in affordances:
                shown = a
                nl_desc = None
                if self.cfg.display_nl and ("<" not in a and ">" not in a):
                    try:
                        from sere.pddl.grounding import parse_grounded
                        name, args = parse_grounded(a)
                        nl_desc = self.nl.act_to_text(name, tuple(args))
                    except Exception:
                        nl_desc = None
                lines.append(self._inline(shown, nl_desc))
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

        # Only include the footer when explicitly enabled
        footer_block = tail if self.cfg.show_footer else ""

        return "\n\n".join(
            p for p in [
                header,
                msg_txt,
                state_txt,
                goal_txt,
                fl_txt,
                aff_txt,
                footer_block,
            ] if p
        ).strip()



    # ---------- Affordances ----------
    def generate_affordances(
        self,
        world: WorldState,
        static_facts: Set[Predicate],
        enable_numeric: bool = True,
    ) -> List[str]:
        """
        Generate affordances against the *scoped* world view:
        - If the action has NO 'number' params -> fully grounded affordances.
        - If the action HAS 'number' params   -> use '<n>' placeholder(s),
        and skip *numeric* preconditions while still checking symbolic ones.
        """
        if not self.cfg.show_affordances:
            return []

        # Scope once
        scoped_world, scoped_static, _room = self._scoped_view(world, static_facts)

        # Ensure object registry is complete before pooling by type
        self._ensure_object_registry(scoped_world, scoped_static)

        # --- helper: type membership that tolerates multi-typed objects ---
        def _is_type(sym: str, typ: str) -> bool:
            t = scoped_world.objects.get(sym)
            if isinstance(t, str):
                return t == typ
            if isinstance(t, (set, list, tuple)):
                return typ in t
            return False

        # Build pools per *declared* param type using the world objects.
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
            # Identify numeric params
            num_positions = [i for i, (_v, typ) in enumerate(act.params) if str(typ).lower() == "number"]

            # Construct candidate pools
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

            for args in combos:
                bind = {var: val for (var, _), val in zip(act.params, args)}

                # Precondition check
                has_number = len(num_positions) > 0
                numeric_ok = enable_numeric and (not has_number)

                ok = True
                for pre in (act.pre or []):
                    # Explicitly skip numeric guards when templating <n>
                    if has_number and self._is_numeric_clause(pre):
                        continue
                    if not eval_clause(scoped_world, scoped_static, pre, bind, enable_numeric=numeric_ok):
                        ok = False
                        break
                if not ok:
                    continue

                # Render: if numeric present, show <n> in those slots; otherwise full args
                shown_args: List[str] = []
                for i, val in enumerate(args):
                    shown_args.append("<n>" if i in num_positions else val)

                if shown_args:
                    afford.append(f"({act.name} {' '.join(shown_args)})")
                else:
                    afford.append(f"({act.name})")

        return afford



    # ---------- Helpers ----------
    def _format_action_catalog(self) -> str:
        rows = []
        for a in sorted(self.domain.actions.values(), key=lambda x: x.name):
            sig = f"{a.name}(" + ", ".join(f"{v}:{t}" for v, t in a.params) + ")"
            # Use NLMapper but preserve placeholders by passing "{var}" as the value
            dummy_args = tuple(f"{{{v}}}" for (v, _t) in a.params)
            catalog_nl = self.nl.act_to_text(a.name, dummy_args)
            energy = self._energy_hint(a)
            dur = self._duration_hint(a)
            rows.append(f"- {sig} — {catalog_nl}  [{energy}; {dur}]")
        return "\n".join(rows)


    # ---------- NL helpers for actions ----------
    def _pick_action_nl(self, action) -> str:
        """
        Choose an NL template for an action. Deterministic: first variant.
        ActionSpec.nl is a List[str] after DomainSpec change, but we tolerate str for BC.
        """
        nl = getattr(action, "nl", None)
        if isinstance(nl, list) and nl:
            return str(nl[0])
        if isinstance(nl, str):
            return nl
        return action.name  # ultra-safe fallback

    def _render_action_nl(self, action, args: List[str]) -> Optional[str]:
        """
        Render a chosen NL template with grounded args.
        We map param var names -> grounded values and .format(**mapping).
        Returns None on any error (caller will omit NL).
        """
        try:
            tmpl = self._pick_action_nl(action)
            var_order = [v for (v, _t) in getattr(action, "params", [])]
            mapping = {v: a for v, a in zip(var_order, args)}
            # Also allow a common alias 'n' for numeric slots if present
            if "n" in tmpl and "n" not in mapping and len(var_order) == len(args):
                for v in var_order:
                    if v.lower() == "n":
                        mapping["n"] = mapping.get(v)
                        break
            return tmpl.format(**mapping)
        except Exception:
            return None


    # ---------- Success goal rendering from termination_rules ----------
    def _render_success_goal_line(self, termination_rules: Optional[List[dict]]) -> str:
        """
        If termination_rules contains a rule with outcome=='success', render a single
        Goal line describing that rule. Supports:
          - when: "<pddl-literal>"
          - when: { any: ["<lit>", "<lit2>", ...] }  -> "lit or lit2"
          - when: { all: ["<lit>", "<lit2>", ...] }  -> "lit and lit2"
        Returns "" if nothing to render.
        """
        if not self.cfg.show_goal or not termination_rules:
            return ""

        # Find first success rule
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

        # Normalize to (mode, exprs)
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

        # Build PDDL string
        joiner = " or " if mode == "any" else (" and " if mode == "all" else "")
        pddl_joined = joiner.join(exprs)

        # Try to build NL for simple literals only; if any fail, omit NL
        nl_parts: List[str] = []
        if self.cfg.display_nl:
            try:
                from sere.pddl.grounding import parse_grounded
                for s in exprs:
                    n, a = parse_grounded(s)  # may raise for non-literals
                    # only map when it's a known predicate
                    if n in self.domain.predicates:
                        nl_parts.append(self.nl.pred_to_text((n, a)))
                    else:
                        nl_parts = []
                        break
                nl_joined = joiner.join(nl_parts) if nl_parts else None
            except Exception:
                nl_joined = None
        else:
            nl_joined = None
        return self._inline(pddl_joined, nl_joined)

    # ---------- Cost/introspection helpers ----------
    _NUM_EFF_RX = re.compile(
        r"^\(\s*(increase|decrease|assign)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+([^\s()]+)\s*\)$"
    )

    def _energy_hint(self, act) -> str:
        """Summarize energy effect for an action.

        Preference order:
        1) Outcome named 'success' -> its num_eff
        2) First outcome (if any)
        3) Action-level num_eff
        4) Default 0.0

        Supports OutcomeSpec objects or plain dicts.
        """

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
            symbolic = []  # keep last symbolic term for display if needed

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

        # Outcomes may be objects or dicts
        outcomes = list(getattr(act, "outcomes", []) or [])

        # 1) Prefer an outcome with status 'success'
        chosen = None
        for oc in outcomes:
            if _outcome_status(oc) == "success":
                chosen = oc
                break

        # 2) Otherwise, use the first outcome if present
        if chosen is None and outcomes:
            chosen = outcomes[0]

        # 3) Pull num_eff from the chosen outcome, else fall back to action-level
        num_eff = _outcome_num_eff(chosen)
        if not num_eff:
            num_eff = _action_num_eff(act)

        return render_delta(num_eff)



    def _duration_hint(self, act) -> str:
        """Summarize action duration in human-readable form."""
        vname = getattr(act, "duration_var", None)
        unit  = getattr(act, "duration_unit", None)
        dur   = getattr(act, "duration", None)
        if isinstance(vname, str) and isinstance(unit, (int, float)):
            # e.g., 0.5 × n
            # keep concise; avoid trailing .0 when possible
            u = f"{unit:.3f}".rstrip("0").rstrip(".")
            return f"duration {u}×{vname}"
        if isinstance(dur, (int, float)):
            d = f"{float(dur):.3f}".rstrip("0").rstrip(".")
            return f"duration {d}"
        # Unknown/default (engine will use env.default_duration)
        return "duration default"
