from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, List
from ..pddl.domain_spec import DomainSpec, Predicate
from ..pddl.nl_mapper import NLMapper
from .world_state import WorldState
from .semantics import eval_clause
import fnmatch


@dataclass
class PromptFormatterConfig:
    # Unified rendering: system prompt + observations
    display_nl: bool = True   # True → NL+PDDL; False → PDDL-only

    # Visibility / limits
    show_briefing: bool = True
    show_objects_in_sysprompt: bool = True
    nl_max_facts: int = 200

    # What to show in obs
    show_goal: bool = True
    show_affordances: bool = True
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
        self.nl = NLMapper(domain)
    
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


    def build_system_prompt(self, *, world: WorldState, time_limit: float | None = None) -> str:
        if not self.cfg.show_briefing:
            return ""

        # Detect whether energy is actually modeled in this instance
        has_energy = any(name == "energy" for (name, _args) in world.fluents.keys())

        # Detect whether time/durations matter by inspecting the domain's actions
        has_time = False
        for a in self.domain.actions.values():
            if (a.duration is not None) or (a.duration_var is not None):
                has_time = True
                break

        parts: list[str] = []

        # -------- High-level explainer (plain text, redundant on purpose) --------
        if has_energy or has_time:
            expl = []
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
                    "Each action takes time. The header shows Time: value. "
                    + (
                        f"This task has a strict time limit of {time_limit:.2f}. "
                        "If the total time exceeds this value the task fails. "
                        if time_limit is not None else
                        "Some tasks may set a maximum time limit. "
                        "If total time exceeds the limit the task fails. "
                    )
                    + "Choose efficient actions to stay within the time limit."
                )
            expl.append(
                "The header always shows the current step number and the maximum allowed steps. "
                "If time is enabled it also shows total elapsed time. If energy is modeled it shows the energy level "
                "for each robot, and the battery capacity if defined."
            )
            expl.append(
                "You must output exactly one valid action per step, formatted as <move>(action arg1 arg2 ...)</move>. "
                "Do not output explanations or multiple actions in one step. Think step by step. "
                "Watch energy and recharge if needed. Watch time and avoid wasted moves."
            )
            parts.append(" ".join(expl))

        # -------- Action catalog (always included) --------
        actions = self._format_action_catalog()  # always PDDL signatures
        parts += [
            f"You are controlling the robot in the '{self.domain.name}' domain.",
            "Reply with exactly one action inside <move>…</move>, e.g., <move>(move r1 A B)</move>.",
            "",
            "Actions (signature — description):",
            actions or "(none)",
        ]

        # -------- Objects listing --------
        if self.cfg.show_objects_in_sysprompt:
            parts += ["", "Objects (PDDL):", self._format_objects_pddl(world)]
            if self.cfg.display_nl:
                parts += ["", "Objects (NL):", self._format_objects_nl(world)]

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
    ) -> str:
        # ----- State (compact inline: PDDL – NL) -----
        facts = [fa for fa in sorted(world.facts) if fa[0] != "adjacent"]
        state_lines: List[str] = []
        for pred in facts:
            pddl = self._pddl(pred)
            nl = self._nl_pred(pred) if self.cfg.display_nl else None
            state_lines.append(self._inline(pddl, nl))
        state_txt = "State:\n" + ("\n".join(state_lines) if state_lines else "(none)")

        # ----- Goal (derived from termination_rules only) -----
        goal_txt = ""
        success_line = self._render_success_goal_line(termination_rules)
        if success_line:
            goal_txt = "Goal:\n" + success_line

        # ----- Fluents (PDDL only, compact) -----
        fl_txt = ""
        if self.cfg.show_fluents and world.fluents:
            prec = max(0, int(self.cfg.fluents_precision))
            rows = []
            for (name, args), val in sorted(world.fluents.items()):
                if not any(fnmatch.fnmatch(name, pat) for pat in (self.cfg.visible_fluents or ["*"])):
                    continue
                argstr = "" if not args else " " + " ".join(args)
                rows.append(f"- (= ({name}{argstr}) {val:.{prec}f})")
            fl_txt = "Fluents:\n" + ("\n".join(rows) if rows else "")

        # ----- Affordances (compact inline: PDDL – NL) -----
        aff_txt = ""
        if self.cfg.show_affordances and affordances:
            from ..pddl.grounding import parse_grounded
            lines = []
            for a in affordances:
                nl = self.nl.act_to_text(*parse_grounded(a)) if self.cfg.display_nl else None
                lines.append(self._inline(a, nl))
            aff_txt = "Valid moves:\n" + "\n".join(lines)

        # ----- Messages (inline, no label) -----
        msg_txt = ""
        if self.cfg.show_messages and messages:
            if self.cfg.messages_inline:
                msg_txt = "\n".join(messages)   # exactly as you asked: plain lines
            else:
                msg_txt = "Messages:\n  - " + "\n  - ".join(messages)

        # ----- Header -----
        limit = "" if time_limit is None else f"/{time_limit:.2f}"
        time_txt = f" | Time: {time_val:.2f}{limit}" if durations_on else ""
        energy_bits = []
        for sym, types in sorted(world.objects.items()):
            if any(t.lower() == "robot" for t in (types or [])):
                e = world.get_fluent("energy", (sym,))
                cap = world.get_fluent("battery-cap", (sym,))
                has_cap = (("battery-cap", (sym,)) in world.fluents)
                energy_bits.append(f"{sym} {e:.2f}/{cap:.2f}" if has_cap else f"{sym} {e:.2f}")
        energy_txt = (" | Energy: " + ", ".join(energy_bits)) if energy_bits else ""
        header = f"Steps: {steps}/{max_steps}{time_txt}{energy_txt}"

        # ----- Stitch: header → message → state → goal → fluents → affordances -----
        return "\n\n".join(
            p for p in [
                header,
                msg_txt,          # moved up, inline
                state_txt,
                goal_txt,
                fl_txt,
                aff_txt,
                "Reply with <move>(action args)</move>."
            ] if p
        ).strip()


    def _format_fluents(
        self,
        *,
        world_fluents: Dict[Tuple[str, Tuple[str, ...]], float],
        prev_fluents: Dict[Tuple[str, Tuple[str, ...]], float],
    ) -> str:
        rows = []
        prec = max(0, int(self.cfg.fluents_precision))
        for (name, args), val in sorted(world_fluents.items()):
            # Time is engine-owned
            if not any(fnmatch.fnmatch(name, pat) for pat in (self.cfg.visible_fluents or ["*"])):
                continue
            key = f"({name}{'' if not args else ' ' + ' '.join(args)})"
            delta_txt = ""
            if self.cfg.show_fluent_deltas:
                prev = prev_fluents.get((name, args))
                if prev is not None:
                    dv = val - prev
                    if abs(dv) >= 10 ** (-prec):
                        sign = "+" if dv >= 0 else ""
                        delta_txt = f" ({sign}{dv:.{prec}f})"
            rows.append(f"{key}={val:.{prec}f}{delta_txt}")
        return ("Fluents: " + ", ".join(rows)) if rows else ""


    # ---------- Affordances ----------
    def generate_affordances(
        self,
        world: WorldState,
        static_facts: Set[Predicate],
        enable_numeric: bool = True,
    ) -> List[str]:
        """
        Produce grounded actions that satisfy an action's preconditions under the
        *current* world + statics. Supports objects with multiple types:
          - world.objects[sym] may be a str (single type) OR an iterable (set/list/tuple)
            of types; we treat membership accordingly.
        Returns a (possibly empty) list. Never returns None.
        """
        if not self.cfg.show_affordances:
            return []

        # --- helper: type membership that tolerates multi-typed objects ---
        def _is_type(sym: str, typ: str) -> bool:
            t = world.objects.get(sym)
            if isinstance(t, str):
                return t == typ
            if isinstance(t, (set, list, tuple)):
                return typ in t
            return False  # unknown/odd entry → not of this type

        # Build pools per *declared* param type using the world objects.
        by_type: Dict[str, List[str]] = {}
        for sym in world.objects.keys():
            t = world.objects[sym]
            if isinstance(t, str):
                by_type.setdefault(t, []).append(sym)
            elif isinstance(t, (set, list, tuple)):
                for tt in t:
                    by_type.setdefault(str(tt), []).append(sym)

        def cartesian(lists: List[List[str]]) -> List[List[str]]:
            res = [[]]
            for lst in lists:
                # guard against empty pools to avoid exploding combinations
                if not lst:
                    return []
                res = [r + [x] for r in res for x in lst]
            return res

        afford: List[str] = []
        for act in sorted(self.domain.actions.values(), key=lambda x: x.name):
            # Construct candidate pools in the order of act.params
            pools: List[List[str]] = []
            for _, typ in act.params:
                # Prefer the fast lookup from by_type; fall back to scanning for safety
                pool = by_type.get(typ)
                if pool is None:
                    pool = [s for s in world.objects if _is_type(s, typ)]
                pools.append(pool)

            combos = cartesian(pools)
            if not combos:
                continue

            for args in combos:
                bind = {var: val for (var, _), val in zip(act.params, args)}

                # Precondition check (supports numeric, (not ...), (or ...), etc.)
                ok = True
                for pre in act.pre or []:
                    if not eval_clause(world, static_facts, pre, bind, enable_numeric=enable_numeric):
                        ok = False
                        break
                if not ok:
                    continue

                afford.append(f"({act.name} {' '.join(args)})")

        return afford


    # ---------- Helpers ----------
    def _format_action_catalog(self) -> str:
        rows = []
        for a in sorted(self.domain.actions.values(), key=lambda x: x.name):
            sig = f"{a.name}(" + ", ".join(f"{v}:{t}" for v, t in a.params) + ")"
            rows.append(f"- {sig} — {a.nl}")
        return "\n".join(rows)

    def _format_objects_nl(self, world: WorldState) -> str:
        by_type: Dict[str, List[str]] = {}
        for sym, types in sorted(world.objects.items()):
            for t in sorted(types or []):
                by_type.setdefault(t, []).append(sym)
        return "\n".join(
            f"- {t}: {', '.join(sorted(syms))}"
            for t, syms in sorted(by_type.items())
        ) or "(none)"


    def _format_objects_pddl(self, world: WorldState) -> str:
        # Group symbols under every type they claim
        by_type: Dict[str, List[str]] = {}
        for sym, types in sorted(world.objects.items()):
            for t in sorted(types or []):
                by_type.setdefault(t, []).append(sym)
        chunks = [f"{' '.join(sorted(syms))} - {t}" for t, syms in sorted(by_type.items())]
        return "(:objects " + " ".join(chunks) + ")"


    def _format_state_nl(self, facts: Set[Predicate]) -> str:
        lines: List[str] = []
        for i, (p, a) in enumerate(sorted(facts)):
            if p == "adjacent":
                continue
            if i >= self.cfg.nl_max_facts:
                lines.append("... (truncated)")
                break
            try:
                lines.append(self.nl.pred_to_text((p, a)))
            except Exception:
                lines.append(f"({p} {' '.join(a)})")
        return "\n".join(f"- {x}" for x in lines) if lines else "(none)"

    def _format_goal_nl(self, goal: List[Predicate]) -> str:
        out: List[str] = []
        for (name, args) in goal:
            spec = self.domain.predicates.get(name)
            if spec:
                try:
                    mapping = {spec.args[i][0]: args[i] for i in range(len(spec.args))}
                    out.append(spec.nl.format(**mapping))
                    continue
                except Exception:
                    pass
            out.append(f"({name} {' '.join(args)})")
        return "\n".join(f"- {x}" for x in out) if out else ""

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
                from ..pddl.grounding import parse_grounded
                for s in exprs:
                    n, a = parse_grounded(s)  # may raise for non-literals
                    # only map when it's a known predicate
                    if n in self.domain.predicates:
                        nl_parts.append(self.nl.pred_to_text((n, a)))
                    else:
                        nl_parts = []
                        break
                if nl_parts:
                    nl_joined = joiner.join(nl_parts)
                else:
                    nl_joined = None
            except Exception:
                nl_joined = None
        else:
            nl_joined = None

        return self._inline(pddl_joined, nl_joined)
