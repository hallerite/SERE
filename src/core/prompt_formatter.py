# src/core/prompt_formatter.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
from ..pddl.domain_spec import DomainSpec, Predicate
from ..pddl.nl_mapper import NLMapper
from .world_state import WorldState
from .semantics import eval_clause
import fnmatch

@dataclass
class PromptFormatterConfig:
    # briefing/system-prompt
    show_briefing: bool = True
    include_static_in_briefing: bool = True
    include_problem_pddl_in_briefing: bool = True

    # observation formatting
    obs_mode: str = "nl"           # "nl" | "pddl" | "both"
    nl_max_facts: int = 200
    show_goal_nl: bool = True

    # affordances
    show_affordances: bool = True

    # fluents/messages
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


    # ---------- Briefing ----------
    def build_system_prompt(self, world: WorldState, static_facts: Set[Predicate], goal: List[Predicate]) -> str:
        if not self.cfg.show_briefing:
            return ""

        by_type: Dict[str, List[str]] = {}
        for sym, typ in sorted(world.objects.items()):
            by_type.setdefault(typ, []).append(sym)
        obj_lines = "\n".join(f"- {t}: {', '.join(sorted(syms))}" for t, syms in sorted(by_type.items()))

        statics = self._format_static_facts_nl(static_facts) if self.cfg.include_static_in_briefing else ""
        actions = self._format_action_catalog()
        init_nl = self._format_state_nl(world.facts)
        goal_nl = self._format_goal_nl(goal)
        pddl = world.to_problem_pddl("instance", static_facts, goal) if self.cfg.include_problem_pddl_in_briefing else ""

        parts = [
            f"You are controlling the robot in the '{self.domain.name}' domain.",
            "Reply with exactly one action inside <move>…</move>, e.g., <move>(move r1 A B)</move>.",
            "",
            "Objects by type:",
            obj_lines or "(none)",
            "",
            "Actions (signature — description):",
            actions or "(none)",
        ]
        if self.cfg.include_static_in_briefing:
            parts += ["", "Static facts:", statics or "(none)"]
        parts += ["", "Initial state (natural language):", init_nl or "(none)"]
        if self.cfg.show_goal_nl:
            parts += ["", "Goal (natural language):", goal_nl or "(none)"]
        if pddl:
            parts += ["", "PDDL problem:", pddl]
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
        prev_fluents: Dict[Tuple[str, Tuple[str,...]], float],
        affordances: List[str],
    ) -> str:
        # state
        facts_nl = self._format_state_nl(world.facts)
        state_pddl = "\n  ".join(f"({p} {' '.join(a)})" for (p,a) in sorted(world.facts) if p != "adjacent")

        body = []
        mode = (self.cfg.obs_mode or "nl").lower()
        if mode in ("nl", "both"):
            body.append("State (NL):\n" + facts_nl)
        if mode in ("pddl", "both"):
            body.append("State (PDDL):\n  " + state_pddl)

        # time/steps
        time_txt = f" | Time: {time_val:.2f}" if durations_on else ""
        header = f"Steps: {steps}/{max_steps}{time_txt}"

        # goal
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
            if not any(fnmatch.fnmatch(name, pat) for pat in (self.cfg.visible_fluents or ["*"])):
                continue
            key = f"({name}{'' if not args else ' ' + ' '.join(args)})"
            delta_txt = ""
            if self.cfg.show_fluent_deltas:
                prev = prev_fluents.get((name, args))
                if prev is not None:
                    dv = val - prev
                    # avoid microscopic noise
                    if abs(dv) >= 10 ** (-prec):
                        sign = "+" if dv >= 0 else ""
                        delta_txt = f" ({sign}{dv:.{prec}f})"
            rows.append(f"{key}={val:.{prec}f}{delta_txt}")
        return ("Fluents: " + ", ".join(rows)) if rows else ""

    # ---------- Affordances ----------
    def generate_affordances(self, world: WorldState, static_facts: Set[Predicate]) -> List[str]:
        if not self.cfg.show_affordances:
            return []
        # Full grounding across typed pools (small domains => acceptable)
        by_type: Dict[str, List[str]] = {}
        for sym, typ in world.objects.items():
            by_type.setdefault(typ, []).append(sym)

        def cartesian(lists: List[List[str]]) -> List[List[str]]:
            res = [[]]
            for lst in lists:
                res = [r+[x] for r in res for x in lst]
            return res

        afford: List[str] = []
        for act in sorted(self.domain.actions.values(), key=lambda x: x.name):
            pools = [by_type.get(typ, []) for _, typ in act.params]
            if any(not pool for pool in pools):
                continue
            for args in cartesian(pools):
                bind = {var: val for (var,_), val in zip(act.params, args)}
                # every pre must evaluate true
                ok = True
                for pre in act.pre:
                    if not eval_clause(world, static_facts, pre, bind, enable_numeric=True):
                        ok = False; break
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

    def _format_static_facts_nl(self, static_facts: Set[Predicate]) -> str:
        lines: List[str] = []
        adj: Dict[str, List[str]] = {}
        for (p,a) in static_facts:
            if p == "adjacent" and len(a) == 2:
                adj.setdefault(a[0], []).append(a[1])
            else:
                try:
                    lines.append(self.nl.pred_to_text((p,a)))
                except Exception:
                    lines.append(f"({p} {' '.join(a)})")
        if adj:
            lines.append("Adjacency:")
            for l, nbrs in sorted(adj.items()):
                lines.append(f"- {l} ↔ {', '.join(sorted(nbrs))}")
        return "\n".join(lines)

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
