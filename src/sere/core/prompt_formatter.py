from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
from ..pddl.domain_spec import DomainSpec, Predicate
from ..pddl.nl_mapper import NLMapper
from .world_state import WorldState
from .semantics import eval_clause
import fnmatch
import re


@dataclass
class PromptFormatterConfig:
    # Unified rendering: system prompt + observations
    display_nl: bool = True   # True → NL+PDDL; False → PDDL-only

    # Visibility / limits
    show_briefing: bool = True
    show_objects_in_sysprompt: bool = True

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

    def _format_objects_name_type(self, world: WorldState) -> str:
        """
        Render objects as 'name: {type1, type2}'.
        Example:
        urn: {appliance, container}
        bot: {robot}
        """
        lines = []
        for sym, types in sorted(world.objects.items()):
            type_str = ", ".join(sorted(types)) if types else "untyped"
            lines.append(f"{sym}: {{{type_str}}}")
        return "\n".join(lines) if lines else "(none)"

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
            "Actions (signature — description  [costs]):",
            actions or "(none)",
        ]

        # -------- Objects listing --------
        if self.cfg.show_objects_in_sysprompt:
            parts += ["", "Objects (name - type):", self._format_objects_name_type(world)]

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
            lines = []
            for a in affordances:
                shown = a  # generate_affordances already returns "(act args)" — single parens
                nl_desc = None
                if self.cfg.display_nl and ("<" not in a and ">" not in a):
                    try:
                        from ..pddl.grounding import parse_grounded
                        name, args = parse_grounded(a)  # -> ("move", ["r1","A","B"])
                        action = self.domain.actions.get(name)
                        if action:
                            nl_desc = self._render_action_nl(action, args)
                    except Exception:
                        nl_desc = None
                lines.append(self._inline(shown, nl_desc))
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


    # ---------- Affordances ----------
    def generate_affordances(
        self,
        world: WorldState,
        static_facts: Set[Predicate],
        enable_numeric: bool = True,
    ) -> List[str]:
        """
        Mixed strategy:
        - Actions with NO 'number' params -> fully grounded affordances (as before).
        - Actions WITH 'number' params   -> template affordances:
                use '<n>' for numeric slots, ground object params, and
                check only non-numeric preconditions (numeric checks are skipped).
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
            # Identify numeric params
            num_positions = [i for i, (_v, typ) in enumerate(act.params) if typ.lower() == "number"]

            # Construct candidate pools
            pools: List[List[str]] = []
            for i, (_var, typ) in enumerate(act.params):
                if typ.lower() == "number":
                    # Use a single placeholder candidate; we won't enumerate numbers
                    pools.append(["<n>"])
                else:
                    pool = by_type.get(typ)
                    if pool is None:
                        pool = [s for s in world.objects if _is_type(s, typ)]
                    pools.append(pool)

            combos = cartesian(pools)
            if not combos:
                continue

            for args in combos:
                bind = {var: val for (var, _), val in zip(act.params, args)}

                # Precondition check:
                # - For templated (has number): skip numeric evaluation (can't bind <n>)
                # - For fully grounded (no number): use engine's numeric setting
                has_number = len(num_positions) > 0
                numeric_ok = enable_numeric and (not has_number)

                ok = True
                for pre in (act.pre or []):
                    if not eval_clause(world, static_facts, pre, bind, enable_numeric=numeric_ok):
                        ok = False
                        break
                if not ok:
                    continue

                # Render: if numeric present, show <n> in those slots; otherwise full args
                shown_args: List[str] = []
                for i, val in enumerate(args):
                    if i in num_positions:
                        shown_args.append("<n>")
                    else:
                        shown_args.append(val)

                afford.append(f"({act.name} {' '.join(shown_args)})")

        return afford


    # ---------- Helpers ----------
    def _format_action_catalog(self) -> str:
        rows = []
        for a in sorted(self.domain.actions.values(), key=lambda x: x.name):
            sig = f"{a.name}(" + ", ".join(f"{v}:{t}" for v, t in a.params) + ")"
            energy = self._energy_hint(a)
            dur = self._duration_hint(a)
            nl_text = self._pick_action_nl(a)
            rows.append(f"- {sig} — {nl_text}  [{energy}; {dur}]")

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

        def _outcome_name(oc) -> str:
            if oc is None:
                return ""
            if isinstance(oc, dict):
                return str(oc.get("name", "")).lower()
            return str(getattr(oc, "name", "")).lower()

        def _outcome_num_eff(oc):
            if oc is None:
                return []
            if isinstance(oc, dict):
                return oc.get("num_eff") or []
            return getattr(oc, "num_eff", None) or []

        def _action_num_eff(a):
            return getattr(a, "num_eff", None) or []

        def render_delta(num_eff_list) -> str:
            total = 0.0
            symbolic = []  # keep last symbolic term for display if needed

            for expr in (num_eff_list or []):
                s = str(expr).strip()
                m = self._NUM_EFF_RX.match(s)
                if not m:
                    continue
                op, fname, _argstr, rhs = m.groups()
                if fname != "energy":
                    continue

                if op == "assign":
                    return f"energy \u2192 {rhs.strip()}"

                rhs = rhs.strip()
                try:
                    val = float(rhs)
                    if op == "increase":
                        total += val
                    elif op == "decrease":
                        total -= val
                except Exception:
                    symbolic.append((op, rhs))

            if symbolic and abs(total) < 1e-12:
                op, rhs = symbolic[-1]
                if op == "increase":
                    return f"energy +{rhs}"
                if op == "decrease":
                    return f"energy -{rhs}"
                if op == "assign":
                    return f"energy \u2192 {rhs}"

            sign = "+" if total >= 0 else ""
            return f"energy {sign}{total:.3f}"

        # Outcomes may be objects or dicts
        outcomes = list(getattr(act, "outcomes", []) or [])

        # 1) Prefer an outcome literally named 'success'
        chosen = None
        for oc in outcomes:
            if _outcome_name(oc) == "success":
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
