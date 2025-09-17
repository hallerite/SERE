from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
from ..pddl.domain_spec import DomainSpec, Predicate
from ..pddl.nl_mapper import NLMapper
from .world_state import WorldState
from .semantics import eval_clause
import fnmatch


@dataclass
class PromptFormatterConfig:
    # System prompt (domain + objects only)
    show_briefing: bool = True
    sysprompt_mode: str = "nl"      # "nl" | "pddl" | "both"
    show_objects_in_sysprompt: bool = True

    # Observation formatting (state/goal/messages/etc.)
    obs_mode: str = "nl"            # "nl" | "pddl" | "both"
    nl_max_facts: int = 200
    show_goal_nl: bool = True

    # Affordances
    show_affordances: bool = True

    # Fluents/messages
    show_fluents: bool = True
    fluents_precision: int = 2
    show_fluent_deltas: bool = True
    visible_fluents: Optional[List[str]] = None   # glob patterns or None -> ["*"]
    show_messages: bool = True


class PromptFormatter:
    def __init__(self, domain: DomainSpec, config: Optional[PromptFormatterConfig] = None):
        self.domain = domain
        self.nl = NLMapper(domain)
        self.cfg = config or PromptFormatterConfig()
        if self.cfg.visible_fluents is None:
            self.cfg.visible_fluents = ["*"]

    # ---------- System Prompt (DOMAIN + OBJECTS ONLY) ----------
    def build_system_prompt(self, *, world: WorldState) -> str:
        """
        Includes:
          - Domain name
          - I/O contract (<move>…</move>)
          - Action catalog
          - Objects in scene (format controlled by sysprompt_mode)
        Excludes:
          - features, static facts, initial state, goal, problem PDDL
        """
        if not self.cfg.show_briefing:
            return ""

        actions = self._format_action_catalog()

        parts = [
            f"You are controlling the robot in the '{self.domain.name}' domain.",
            "Reply with exactly one action inside <move>…</move>, e.g., <move>(move r1 A B)</move>.",
            "",
            "Actions (signature — description):",
            actions or "(none)",
        ]

        if self.cfg.show_objects_in_sysprompt:
            mode = (self.cfg.sysprompt_mode or "nl").lower()
            if mode in ("nl", "both"):
                parts += ["", "Objects by type:", self._format_objects_nl(world)]
            if mode in ("pddl", "both"):
                parts += ["", "Objects (PDDL):", self._format_objects_pddl(world)]

        return "\n".join(parts)

    # ---------- Observation ----------
    def format_obs(
        self,
        *,
        world: WorldState,
        static_facts: Set[Predicate],
        goal: List[Predicate],
        steps: int,
        max_steps: int,
        time_val: float,
        durations_on: bool,
        messages: List[str],
        prev_fluents: Dict[Tuple[str, Tuple[str, ...]], float],
        affordances: List[str],
    ) -> str:
        # state
        facts_nl = self._format_state_nl(world.facts)
        state_pddl = "\n  ".join(
            f"({p} {' '.join(a)})" for (p, a) in sorted(world.facts) if p != "adjacent"
        )

        body = []
        mode = (self.cfg.obs_mode or "nl").lower()
        if mode in ("nl", "both"):
            body.append("State (NL):\n" + facts_nl)
        if mode in ("pddl", "both"):
            body.append("State (PDDL):\n  " + state_pddl)

        # time/steps
        time_txt = f" | Time: {time_val:.2f}" if durations_on else ""
        header = f"Steps: {steps}/{max_steps}{time_txt}"

        # goal lives in OBS (not in system prompt)
        goal_txt = ""
        if self.cfg.show_goal_nl:
            goal_txt = "Goal (NL):\n" + self._format_goal_nl(goal)

        # messages
        msg_txt = ""
        if self.cfg.show_messages and messages:
            msg_txt = "\nMessages:\n  - " + "\n  - ".join(messages)

        # fluents
        fl_txt = ""
        if self.cfg.show_fluents and world.fluents:
            fl_txt = self._format_fluents(
                world_fluents=world.fluents,
                prev_fluents=prev_fluents,
            )

        # affordances
        aff_txt = ""
        if self.cfg.show_affordances and affordances:
            aff_txt = "Valid moves:\n- " + "\n- ".join(affordances)

        return "\n".join([
            header,
            "\n".join(body),
            goal_txt,
            msg_txt,
            fl_txt,
            aff_txt,
            "Reply with <move>(action args)</move>."
        ]).strip()

    def _format_fluents(
        self,
        *,
        world_fluents: Dict[Tuple[str, Tuple[str, ...]], float],
        prev_fluents: Dict[Tuple[str, Tuple[str, ...]], float],
    ) -> str:
        rows = []
        prec = max(0, int(self.cfg.fluents_precision))
        for (name, args), val in sorted(world_fluents.items()):
            # Time is engine-owned; never display a stray 'elapsed' fluent if present.
            if name == "elapsed":
                continue
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
        if not self.cfg.show_affordances:
            return []
        by_type: Dict[str, List[str]] = {}
        for sym, typ in world.objects.items():
            by_type.setdefault(typ, []).append(sym)

        def cartesian(lists: List[List[str]]) -> List[List[str]]:
            res = [[]]
            for lst in lists:
                res = [r + [x] for r in res for x in lst]
            return res

        afford: List[str] = []
        for act in sorted(self.domain.actions.values(), key=lambda x: x.name):
            pools = [by_type.get(typ, []) for _, typ in act.params]
            if any(not pool for pool in pools):
                continue
            for args in cartesian(pools):
                bind = {var: val for (var, _), val in zip(act.params, args)}
                ok = True
                for pre in act.pre:
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
        for sym, typ in sorted(world.objects.items()):
            by_type.setdefault(typ, []).append(sym)
        return "\n".join(f"- {t}: {', '.join(sorted(syms))}" for t, syms in sorted(by_type.items())) or "(none)"

    def _format_objects_pddl(self, world: WorldState) -> str:
        # PDDL-style :objects section (one line, grouped by type)
        by_type: Dict[str, List[str]] = {}
        for sym, typ in sorted(world.objects.items()):
            by_type.setdefault(typ, []).append(sym)
        chunks = []
        for t, syms in sorted(by_type.items()):
            chunks.append(f"{' '.join(sorted(syms))} - {t}")
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
