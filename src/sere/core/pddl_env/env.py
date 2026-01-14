import logging
import random
from typing import Any, Dict, List, Optional, Tuple
from sere.pddl.domain_spec import DomainSpec
from sere.pddl.grounding import SExprError
from sere.core.semantics import eval_clause, EvalNode
from sere.core.world_state import WorldState
from sere.core.invariants import InvariantPlugin
from . import planning, rendering
from .run_mode import RunMode
from .prompt_formatter import PromptFormatter, PromptFormatterConfig

logger = logging.getLogger(__name__)


def extract_tag(s: str, tag: str) -> Optional[str]:
    start, end = f"<{tag}>", f"</{tag}>"
    i, j = s.find(start), s.find(end)
    return None if i < 0 or j < 0 else s[i+len(start):j].strip()

class PDDLEnv:
    def __init__(self, domain: DomainSpec, world: WorldState,
        static_facts: set,
        formatter_config: Optional[dict] = None,
        formatter_obj: Optional[PromptFormatter] = None,
        plugins: Optional[List[InvariantPlugin]] = None,
        max_steps: int = 40, step_penalty: float = -0.01,
        invalid_penalty: float = -0.1,
        termination_rules: Optional[List[dict]] = None,
        run_mode: str | RunMode = RunMode.INTERACTIVE,
        *,
        illegal_move_retries: int = 0,
        invalid_retry_penalty: float = 0.0,
        enable_numeric: bool = False,
        enable_conditional: bool = False,
        enable_durations: bool = False,
        enable_stochastic: bool = True, 
        max_messages: int = 8,
        time_limit: Optional[float] = None,
        default_duration: float = 1.0,
        seed: Optional[int] = None,
        reward_shaping: Optional[dict] = None,
        multi_agent: bool = False):

        self.debug = False # must be explicitly overriden

        self.domain, self.world = domain, world
        self.static_facts = static_facts

        if formatter_obj is not None:
            self.formatter = formatter_obj
        else:
            cfg = PromptFormatterConfig()
            if formatter_config:
                for k, v in formatter_config.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                    else:
                        raise AttributeError(f"Unknown prompt config key: {k}")

            # If caller didn't explicitly set NL stochasticity, mirror env's enable_stochastic
            user_set_nl_stoch = bool(formatter_config and "nl_stochastic" in formatter_config)
            if not user_set_nl_stoch:
                cfg.nl_stochastic = bool(enable_stochastic)

            # If caller didn't provide an NL RNG seed, reuse the env seed for reproducibility
            if cfg.nl_rng_seed is None and seed is not None:
                cfg.nl_rng_seed = int(seed)

            cfg.multi_agent = bool(multi_agent)
            self.formatter = PromptFormatter(self.domain, cfg)

        self._system_prompt_cache = ""

        self.plugins = plugins or []
        
        # Normalize run_mode exactly once to Enum
        if isinstance(run_mode, RunMode):
            self.run_mode = run_mode
        elif isinstance(run_mode, str):
            try:
                self.run_mode = RunMode(run_mode.lower())
            except ValueError:
                raise ValueError(f"Invalid run_mode: {run_mode!r}. Use one of: "
                                f"{', '.join(m.value for m in RunMode)}")
        else:
            raise TypeError("run_mode must be a str or RunMode")

        self.max_steps = max_steps
        self.step_penalty, self.invalid_penalty = step_penalty, invalid_penalty
        self.steps = 0
        self.done = False
        self.enable_numeric = enable_numeric
        self.enable_conditional = enable_conditional
        self.enable_durations = enable_durations
        self.enable_stochastic = enable_stochastic
        self.multi_agent = bool(multi_agent)

        self.illegal_move_retries = max(0, int(illegal_move_retries))
        self.invalid_retry_penalty = float(invalid_retry_penalty)
        self._retries_used_this_turn = 0

        self.default_duration = float(default_duration)
        self.time_limit = time_limit
        self.time = 0.0
        
        self.messages: List[str] = []          # full log (kept for info/debug)
        self._step_messages: List[str] = []     # last-turn-only buffer
        self.max_messages = int(max_messages)

        self.rng = random.Random(seed)

        # ----- Reward shaping -----
        rs = reward_shaping or {}
        self.rs_mode = (rs.get("mode") or "instant").lower()
        self.rs_gamma = float(rs.get("gamma", 1.0))
        # store milestones as list of (expr, weight, once)
        self.rs_milestones: List[Tuple[str, float, bool]] = [
            (str(m["expr"]), float(m.get("reward", 0.0)), bool(m.get("once", True)))
            for m in rs.get("milestones", [])
            if m and m.get("expr") is not None
        ]

        # ----- Define Terminal States -----
        self.termination_rules: List[dict] = []
        for r in (termination_rules or []):
            self.termination_rules.append(dict(
                name=str(r.get("name", "term")),
                when=str(r["when"]),
                outcome=str(r.get("outcome", "terminal")),
                reward=float(r.get("reward", 0.0)),
            ))

        self._rs_seen: set = set()      # indices of milestones achieved (instant)
        self._rs_phi_prev: float = 0.0  # potential at previous state

    def reset(self, *, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        self.steps, self.done = 0, False
        self.time = 0.0
        errs = self.world.validate_invariants()
        for pl in self.plugins:
            errs += pl.validate(self.world, self.static_facts)
        if errs:
            raise ValueError(f"Invariant errors: {errs}")

        self._retries_used_this_turn = 0

        self.messages.clear()
        self._step_messages.clear()   # <-- clear last-turn buffer
        self._system_prompt_cache = self.formatter.build_system_prompt(
            world=self.world,
            static_facts=self.static_facts,
            time_limit=self.time_limit,
            run_mode=self.run_mode,
        )


        # reward shaping
        self._rs_seen.clear()
        self._rs_phi_prev = self._rs_phi(self.world)

        obs_text = rendering.obs(self)
        info = {
            "system_prompt": self._system_prompt_cache,
            "features": dict(
                numeric=self.enable_numeric,
                conditional=self.enable_conditional,
                durations=self.enable_durations,
                stochastic=self.enable_stochastic,
            ),
        }
        return obs_text, info



    def step(self, text: str):
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")
        try:
            plan = planning.parse_actions(text)
        except Exception as e:
            return self._illegal(f"{e}", {})

        # Mode guard: INTERACTIVE must be exactly one action (or one per robot in multi-agent)
        if self.run_mode == RunMode.INTERACTIVE:
            if self.multi_agent:
                n = len(self._robots())
                if len(plan) != n:
                    return self._illegal(
                        f"Interactive multi-agent expects {n} actions (one per robot); got {len(plan)}.",
                        {},
                    )
            elif len(plan) != 1:
                return self._illegal(f"Interactive mode expects exactly one action; got {len(plan)}.", {})

        if self.multi_agent:
            obs, rew, done, info = planning.execute_joint(self, plan)
        else:
            obs, rew, done, info = planning.execute_plan(self, plan, atomic=False)

        # Normalize/annotate info
        info = dict(info)
        info.setdefault("plan_mode", self.run_mode.value)

        if info.get("invalid_move"):
            trace = info.get("plan_trace") or []
            bad = next((t for t in trace if t.get("outcome") == "invalid_move"), None)
            if bad:
                info["plan_aborted"] = True
                info["aborted_at"] = int(bad.get("i", 0))
                info["abort_error"] = bad.get("error") or info.get("error")

        return obs, rew, self.done or done, info


    # ---- helpers (kept on the env for simplicity) ----

    def _robots(self) -> list[str]:
        robots = []
        for sym, types in self.world.objects.items():
            if any(t.lower() == "robot" for t in (types or [])):
                robots.append(sym)
        return robots

    def _robot_loc(self, r: str) -> Optional[str]:
        locs = [a[1] for (pred, a) in self.world.facts if pred == "at" and len(a) == 2 and a[0] == r]
        return locs[0] if len(locs) == 1 else None

    def _has_energy_support(self) -> bool:
        return ("energy" in (self.domain.fluents or {})) and bool(self._robots())

    def _energy_val(self, r: str) -> float:
        return float(self.world.get_fluent("energy", (r,)))

    def _can_recharge_now(self) -> bool:
        for r in self._robots():
            l = self._robot_loc(r)
            if l and ("has-charger", (l,)) in self.static_facts:
                return True
        return False

    def _energy_depleted_unrecoverable(self) -> bool:
        # Terminal iff: (a) numeric enabled, (b) energy fluent modeled,
        # (c) all robots have energy <= 0, and (d) no charger at current location.
        if not self.enable_numeric or not self._has_energy_support():
            return False
        robots = self._robots()
        if not robots:
            return False
        all_empty = all(self._energy_val(r) <= 0.0 for r in robots)
        if not all_empty:
            return False
        return not self._can_recharge_now()

    def _advance_time(self, dur: float, info: Dict[str, Any]):
        self.time += float(dur)
        if self.enable_durations and self.time_limit is not None and self.time > self.time_limit:
            self.done = True
            info["outcome"] = "timeout"
            info["reason"] = "time_limit_exceeded"

    def _post_apply_success(self, info: Dict[str, Any]):
        reward = self.step_penalty

        if self.done:
            pass
        else:
            derived_cache: dict = {}
            for r in self.termination_rules:
                try:
                    if self._eval_expr(r["when"], derived_cache=derived_cache):
                        self.done = True
                        info["outcome"] = r.get("outcome", "terminal")
                        info["terminal_rule"] = r.get("name", "term")
                        reward += float(r.get("reward", 0.0))
                        self._dbg(f"[TERM] step={self.steps+1} rule={r.get('name')} "
                                f"outcome={info['outcome']} term_reward={r.get('reward',0)}")
                        break
                except (SExprError, ValueError, KeyError) as e:
                    # Expected: malformed termination rule expressions
                    rule_name = r.get("name", "unnamed")
                    logger.warning(f"Failed to evaluate termination rule '{rule_name}': {e}")
                    continue
                except Exception as e:
                    # Unexpected error in termination rule - this is a bug in the task YAML
                    rule_name = r.get("name", "unnamed")
                    logger.error(f"Unexpected error in termination rule '{rule_name}': {e}", exc_info=True)
                    # Don't silently skip - raise to surface the broken rule
                    raise RuntimeError(f"Broken termination rule '{rule_name}': {e}") from e

            if not self.done and self._energy_depleted_unrecoverable():
                self.done = True
                info["outcome"] = "out_of_energy"
                info["reason"] = "energy_depleted"

            elif not self.done and self.steps >= self.max_steps:
                self.done = True
                info["outcome"] = "timeout"
                info["reason"] = "max_steps_exceeded"

        if self.done and not info.get("outcome_set_by_action", False):
            oc = (info.get("outcome") or "").lower()
            whitelist = {"success", "timeout", "out_of_energy", "invalid_move"}
            if oc not in whitelist:
                if "outcome" in info:
                    info["outcome_raw"] = info["outcome"]
                info["outcome"] = "failed"
                if "terminal_rule" in info and "reason" not in info:
                    info["reason"] = str(info["terminal_rule"])
                info.setdefault("reason", "implicit_fail")

        if not self.done and "outcome" not in info:
            info["outcome"] = "ongoing"

        last_turn = self._step_messages[:]
        obs = rendering.obs(self)
        info["messages"] = last_turn
        return obs, reward, self.done, info

    def _illegal(self, msg, info):
        self._retries_used_this_turn += 1

        info.update({
            "invalid_move": True,
            "error": msg,
            "outcome": "invalid_move",
            "messages": self.messages[-self.max_messages:],
        })

        exhausted = (self._retries_used_this_turn > self.illegal_move_retries)
        if exhausted:
            self.done = True
            return f"Invalid: {msg}\nGame Over.", self.invalid_penalty, True, info

        retries_left = max(self.illegal_move_retries - self._retries_used_this_turn + 1, 0)
        retry_hdr = (
            f"Invalid: {msg}\n"
            f"Please try again. Retries left this turn: {retries_left}\n\n"
        )
        return retry_hdr + rendering.obs(self), self.invalid_retry_penalty, False, info

    def _eval_expr(self, s: str, *, derived_cache: Optional[dict] = None) -> bool:
        try:
            return eval_clause(
                self.world,
                self.static_facts,
                s,
                {},
                enable_numeric=self.enable_numeric,
                derived_cache=derived_cache,
            )
        except (SExprError, ValueError, KeyError) as e:
            # Expected: malformed expressions in task definitions
            logger.debug(f"Failed to evaluate expression '{s}': {e}")
            return False
        except Exception as e:
            # Unexpected error - this indicates a bug in eval_clause
            logger.error(f"Unexpected error evaluating expression '{s}': {e}", exc_info=True)
            raise

    def _rs_phi(self, world: WorldState) -> float:
        derived_cache: dict = {}
        return sum(w for (expr, w, _) in self.rs_milestones if self._eval_expr(expr, derived_cache=derived_cache))

    def _energy_cap(self, r: str) -> float | None:
        if not self.enable_numeric:
            return None
        cap = self.world.get_fluent("battery-cap", (r,))
        return cap if (cap > 0.0 or ("battery-cap", (r,)) in self.world.fluents) else None

    def _enforce_energy_bounds(self):
        if not self.enable_numeric:
            return
        for r in self._robots():
            e = self.world.get_fluent("energy", (r,))
            if e < 0.0:
                self.world.set_fluent("energy", (r,), 0.0)
                continue
            cap = self._energy_cap(r)
            if cap is not None and e > cap:
                self.world.set_fluent("energy", (r,), cap)

    def _format_invalid_report(self, act_name: str, args: Tuple[str, ...], failures: List[EvalNode]) -> str:
        from sere.pddl.nl_mapper import NLMapper
        nl = NLMapper(self.domain)

        def _pred_nl(grounded: str) -> Optional[str]:
            try:
                from sere.pddl.grounding import parse_grounded
                name, gargs = parse_grounded(grounded)
                if name in self.domain.predicates:
                    return nl.pred_to_text((name, gargs))
            except (SExprError, ValueError, KeyError):
                # Expected: malformed grounded predicate string
                # Silent failure is OK here - we're just generating helpful NL descriptions
                pass
            return None

        def _fmt_node(n: EvalNode, indent: int = 0) -> List[str]:
            pad = "  " * indent
            lines: List[str] = []
            bullet = f"{pad}- "
            if n.kind == "num":
                grounded = n.grounded or n.expr
                cur = n.details.get("current")
                lines.append(f"{bullet}Required: {grounded}")
                lines.append(f"{pad}    Actually: {n.details.get('fluent')}({', '.join(n.details.get('args') or [])})={cur:.2f}")
            elif n.kind == "distinct":
                grounded = n.grounded or n.expr
                dups = n.details.get("duplicates") or []
                lines.append(f"{bullet}Required: {grounded}")
                if n.satisfied:
                    lines.append(f"{pad}    Actually: OK.")
                else:
                    lines.append(f"{pad}    Actually: duplicates found → {', '.join(dups)}.")
            elif n.kind == "not":
                child = n.children[0] if n.children else None
                if child and child.grounded:
                    ground = child.grounded
                    lines.append(f"{bullet}Required: (not {ground})")
                    maybe = _pred_nl(ground)
                    if maybe:
                        lines.append(f"{pad}    NL: {maybe}")
                    lines.append(f"{pad}    Actually: true.")
                else:
                    lines.append(f"{bullet}Required: {n.expr}")
                    lines.append(f"{pad}    Actually: violated.")
            elif n.kind == "or":
                lines.append(f"{bullet}Required: one of:")
                for c in n.children:
                    g = c.grounded or c.expr
                    lines.append(f"{pad}    • {g}  — {'OK' if c.satisfied else 'false'}")
            else:
                g = n.grounded or n.expr
                nl_line = _pred_nl(g)
                lines.append(f"{bullet}Required: {g}" + (f" — \"{nl_line}\"" if nl_line else ""))
                if n.satisfied:
                    lines.append(f"{pad}    Actually: OK.")
                else:
                    if g.startswith("(co-located"):
                        xs = n.details.get("locs_x", [])
                        ys = n.details.get("locs_y", [])
                        lines.append(f"{pad}    Actually: not co-located. x at {xs or '∅'}, y at {ys or '∅'}.")
                    else:
                        lines.append(f"{pad}    Actually: false.")
            return lines

        head = f"Preconditions were not satisfied for ({act_name} {' '.join(args)}):"
        body_lines: List[str] = []
        for n in failures:
            body_lines.extend(_fmt_node(n))
        return head + "\n\n" + "\n".join(body_lines)

    def _dbg(self, *a):
        if getattr(self, "debug", False):
            print(*a, flush=True)
