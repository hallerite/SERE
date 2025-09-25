from typing import Any, Dict, List, Optional, Tuple
import re, random
from ..pddl.domain_spec import DomainSpec, ActionSpec, Predicate
from ..pddl.grounding import parse_grounded, instantiate, ground_literal
from .prompt_formatter import PromptFormatter, PromptFormatterConfig
from .semantics import eval_num_pre, apply_num_eff, eval_clause, trace_clause, EvalNode
from .world_state import WorldState
from .invariants import InvariantPlugin

def extract_tag(s: str, tag: str) -> Optional[str]:
    start, end = f"<{tag}>", f"</{tag}>"
    i, j = s.find(start), s.find(end)
    return None if i < 0 or j < 0 else s[i+len(start):j].strip()

def _format_msg(template: str, bind: Dict[str, str], world: WorldState) -> str:
    """
    Replace {vars} and ?vars with their bound symbols.
    Also supports simple inline probes:
      - "(quality ?a)" → current numeric fluent value
      - "(pred ?x ...)" → prints "true"/"false"
    Lightweight on purpose to avoid a full parser.
    """
    s = template

    # Curly-brace placeholders: "{r}" → "r1"
    for k, v in bind.items():
        s = s.replace("{" + k + "}", v)

    # Question-mark placeholders: "?r" → "r1"
    for k, v in bind.items():
        s = re.sub(rf'\?{re.escape(k)}(\b|[^A-Za-z0-9_?-])', lambda m: v + (m.group(1) or ''), s)

    # Inline probes of form "(name arg1 arg2)"
    for token in re.findall(r"\([^)]+\)", s):
        try:
            name, args = parse_grounded(token)
            if name in ["<", ">", "<=", ">=", "="]:
                continue
            # try fluent read
            val = world.get_fluent(name, args)
            if val != 0.0 or (name, args) in world.fluents:
                s = s.replace(token, f"{val:.2f}")
            else:
                truth = world.holds((name, args))
                s = s.replace(token, "true" if truth else "false")
        except Exception:
            # leave token as-is if not parseable
            pass
    return s


class PDDLEnv:
    def __init__(self, domain: DomainSpec, world: WorldState,
        static_facts: set,
        formatter_config: Optional[dict] = None,
        formatter_obj: Optional[PromptFormatter] = None,
        plugins: Optional[List[InvariantPlugin]] = None,
        max_steps: int = 40, step_penalty: float = -0.01,
        invalid_penalty: float = -0.1,
        termination_rules: Optional[List[dict]] = None,
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
        reward_shaping: Optional[dict] = None):

        self.domain, self.world = domain, world
        self.static_facts = static_facts

        # formatter
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
            self.formatter = PromptFormatter(self.domain, cfg)
        self._system_prompt_cache = ""

        self.plugins = plugins or []
        self.max_steps = max_steps
        self.step_penalty, self.invalid_penalty = step_penalty, invalid_penalty
        self.steps = 0
        self.done = False
        self.enable_numeric = enable_numeric
        self.enable_conditional = enable_conditional
        self.enable_durations = enable_durations
        self.enable_stochastic = enable_stochastic

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
            time_limit=self.time_limit,
        )

        # reward shaping
        self._rs_seen.clear()
        self._rs_phi_prev = self._rs_phi(self.world)

        obs_text = self._obs()  # PLAIN TEXT ONLY
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

        info: Dict[str, Any] = {}
        move = extract_tag(text, "move")
        if not move:
            return self._illegal("Missing <move>(...)</move>.", info)

        try:
            name, args = parse_grounded(move)
        except Exception as e:
            return self._illegal(f"Parse error: {e}", info)

        if name not in self.domain.actions:
            return self._illegal(f"Unknown action '{name}'", info)

        act: ActionSpec = self.domain.actions[name]

        # ---- Arity check ----
        expected = len(act.params)
        got = len(args)
        if got != expected:
            return self._illegal(f"Arity mismatch for action '{name}': expected {expected}, got {got}", info)
        bind = {var: val for (var, _), val in zip(act.params, args)}
        
        # Validate duration_var if present: must parse as float > 0
        if getattr(act, "duration_var", None):
            n_name = act.duration_var
            raw = bind.get(n_name)
            try:
                dur_multiplier = float(raw)
            except Exception:
                return self._illegal(f"Bad duration_var '{n_name}': {raw!r}", info)
            if dur_multiplier <= 0.0:
                return self._illegal(f"Duration multiplier '{n_name}' must be > 0.", info)

        # ---------- Preconditions ----------
        failures: List[EvalNode] = []
        for s in (act.pre or []):
            node = trace_clause(self.world, self.static_facts, s, bind, enable_numeric=self.enable_numeric)
            if not node.satisfied:
                failures.append(node)
        if failures:
            return self._illegal(self._format_invalid_report(name, args, failures), info)

        # ---------- Apply base effects ----------
        _, add, dele = instantiate(self.domain, act, args)

        def _is_unbound(x: Any) -> bool:
            return isinstance(x, str) and x.startswith("?")

        def _expand_delete_patterns(deletes: List[Predicate], facts: set) -> List[Predicate]:
            expanded: List[Predicate] = []
            for (pred, argtup) in deletes:
                if any(_is_unbound(a) for a in argtup):
                    for (p, a) in list(facts):
                        if p != pred:
                            continue
                        ok = True
                        for i, pat in enumerate(argtup):
                            if _is_unbound(pat):
                                continue
                            if i >= len(a) or pat != a[i]:
                                ok = False
                                break
                        if ok:
                            expanded.append((p, a))
                else:
                    expanded.append((pred, argtup))
            return expanded

        def _robot_sym_from_params(act: ActionSpec, arg_tuple: tuple) -> Optional[str]:
            for i, (_, ty) in enumerate(act.params):
                if ty.lower() == "robot":
                    return arg_tuple[i] if i < len(arg_tuple) else None
            return None

        def _unique_robot_loc(world: WorldState, r: Optional[str]) -> Optional[str]:
            if not r:
                return None
            locs = [a[1] for (pred, a) in world.facts if pred == "at" and len(a) == 2 and a[0] == r]
            return locs[0] if len(locs) == 1 else None

        def _expand_add_patterns(adds: List[Predicate], world: WorldState,
                                act: ActionSpec, arg_tuple: tuple) -> List[Predicate]:
            expanded: List[Predicate] = []
            r_cached: Optional[str] = None
            rloc_cached: Optional[str] = None
            rloc_computed = False

            def _get_rloc() -> Optional[str]:
                nonlocal r_cached, rloc_cached, rloc_computed
                if not rloc_computed:
                    r_cached = _robot_sym_from_params(act, arg_tuple)
                    rloc_cached = _unique_robot_loc(world, r_cached)
                    rloc_computed = True
                return rloc_cached

            for (pred, argtup) in adds:
                if any(_is_unbound(a) for a in argtup):
                    if pred == "obj-at":
                        rloc = _get_rloc()
                        if rloc is None:
                            raise ValueError("Cannot infer location for add; robot has no unique location.")
                        new_args = tuple(rloc if _is_unbound(a) else a for a in argtup)
                        expanded.append((pred, new_args))
                    else:
                        raise ValueError(f"Unbound variable in add for {pred}; not supported.")
                else:
                    expanded.append((pred, argtup))
            return expanded

        dele = _expand_delete_patterns(dele, self.world.facts)
        add  = _expand_add_patterns(add, self.world, act, args)

        self.world.apply(add, dele)

        # ---------- Conditional effects ----------
        if self.enable_conditional and act.cond:
            for cb in act.cond:
                ok = True
                for w in cb.when:
                    if not eval_clause(self.world, self.static_facts, w, bind, enable_numeric=self.enable_numeric):
                        ok = False
                        break
                if ok:
                    for a in cb.add:
                        _, litp = ground_literal(a, bind)
                        self.world.facts.add(litp)
                    for d in cb.delete:
                        _, litp = ground_literal(d, bind)
                        self.world.facts.discard(litp)
                    if self.enable_numeric:
                        for ne in cb.num_eff:
                            apply_num_eff(self.world, ne, bind, info)
                    for m in cb.messages:
                        self._push_msg(_format_msg(m, bind, self.world))

        # ---------- Numeric effects (unconditional) ----------
        if self.enable_numeric and act.num_eff:
            for ne in act.num_eff:
                apply_num_eff(self.world, ne, bind, info)

        # Enforce battery bounds after base numeric updates
        self._enforce_energy_bounds()

        # ---------- Stochastic outcomes ----------
        if getattr(act, "outcomes", None):
            # Filter outcomes by their 'when' guards (only among currently valid branches)
            valid = []
            for oc in act.outcomes:
                ok = True
                for w in (oc.when or []):
                    if not eval_clause(self.world, self.static_facts, w, bind, enable_numeric=self.enable_numeric):
                        ok = False
                        break
                if ok:
                    valid.append(oc)

            choice = None
            roll = None
            totp = None

            if self.enable_stochastic:
                # Sample proportionally to p among valid outcomes
                totp = sum(max(0.0, float(getattr(oc, "p", 0.0))) for oc in valid)
                if valid and (totp is None or totp <= 0.0):
                    info["stochastic_warning"] = "nonpositive_total_probability"
                acc = 0.0
                roll = self.rng.random() * totp if (totp and totp > 0.0) else None
                for oc in valid:
                    acc += max(0.0, float(getattr(oc, "p", 0.0)))
                    if roll is not None and roll <= acc:
                        choice = oc
                        break
                if choice is None and valid:
                    # fallback to last valid if probs degenerate
                    choice = valid[-1]
            else:
                # Deterministic behavior (enable_stochastic == False):
                # 1) If exactly ONE valid outcome, take it (even if it's not named 'success').
                #    This lets failure-only branches like 'spill' fire deterministically.
                # 2) If MULTIPLE valid outcomes, prefer the one explicitly named 'success'.
                #    If none is named 'success', do nothing (mis-specified domain; tests catch it).
                if len(valid) == 1:
                    choice = valid[0]
                else:
                    for oc in valid:
                        if str(getattr(oc, "name", "")).lower() == "success":
                            choice = oc
                            break


            if choice:
                add = []
                dele = []
                for s in (choice.add or []):
                    _, litp = ground_literal(s, bind)
                    add.append(litp)
                for s in (choice.delete or []):
                    _, litp = ground_literal(s, bind)
                    dele.append(litp)
                dele = _expand_delete_patterns(dele, self.world.facts)
                add  = _expand_add_patterns(add, self.world, act, args)
                self.world.apply(add, dele)

                if self.enable_numeric and getattr(choice, "num_eff", None):
                    for ne in (choice.num_eff or []):
                        apply_num_eff(self.world, ne, bind, info)

                for m in (choice.messages or []):
                    self._push_msg(_format_msg(m, bind, self.world))

                # Debug info (present whether stochastic or not)
                info["outcome_branch"] = str(getattr(choice, "name", "chosen"))
                if self.enable_stochastic:
                    info["stochastic_outcome"] = info["outcome_branch"]
                    info["stochastic_roll"] = roll
                    info["stochastic_total_p"] = totp


        # Re-enforce bounds after any stochastic numeric updates
        self._enforce_energy_bounds()

        # ---------- Base action messages (after effects so probes see new state) ----------
        for msg in getattr(act, "messages", []) or []:
            self._push_msg(_format_msg(msg, bind, self.world))            # CHANGED

        # ---------- Time / duration ----------
        if self.enable_durations:
            vname = getattr(act, "duration_var", None)
            unit = getattr(act, "duration_unit", None)

            if isinstance(vname, str) and isinstance(unit, (int, float)):
                sval = bind.get(vname)
                if sval is None:
                    return self._illegal(f"Bad duration var '{vname}' value: {sval!r}", info)
                try:
                    dur = float(unit) * float(sval)
                except (TypeError, ValueError):
                    return self._illegal(f"Bad duration var '{vname}' value: {sval!r}", info)
            else:
                dur = float(act.duration) if (act.duration is not None) else self.default_duration

            self._advance_time(dur, info)

        # ---------- Plugins (post) ----------
        for pl in self.plugins:
            errs = pl.validate(self.world, self.static_facts)
            if errs:
                return self._illegal(f"Postcondition violated: {errs}", info)
        # ---------- Bookkeeping / termination ----------
        self.steps += 1
        obs, base_r, done, info = self._post_apply_success(info)

        # ---------- Reward shaping payout ----------
        rs_bonus = 0.0
        if self.rs_milestones:
            if self.rs_mode == "potential":
                phi_now = self._rs_phi(self.world)
                rs_bonus = self.rs_gamma * phi_now - self._rs_phi_prev
                self._rs_phi_prev = phi_now
            else:  # "instant"
                for i, (expr, w, once) in enumerate(self.rs_milestones):
                    if once and i in self._rs_seen:
                        continue
                    if self._eval_expr(expr):
                        rs_bonus += w
                        if once:
                            self._rs_seen.add(i)
        if abs(rs_bonus) > 0:
            info["shaping_bonus"] = rs_bonus

        self._retries_used_this_turn = 0

        return obs, base_r + rs_bonus, done, info




    # --- helpers ---
        # ----- Energy/outcome helpers -----
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
        # Only consider energy logic if the domain declares the fluent and there is a robot.
        return ("energy" in (self.domain.fluents or {})) and bool(self._robots())

    def _energy_val(self, r: str) -> float:
        # world.get_fluent defaults to 0.0 if unset.
        return float(self.world.get_fluent("energy", (r,)))

    def _can_recharge_now(self) -> bool:
        # Recharge preconditions: (at ?r ?l) and static (has-charger ?l).
        for r in self._robots():
            l = self._robot_loc(r)
            if l and ("has-charger", (l,)) in self.static_facts:
                return True
        return False

    def _energy_depleted_unrecoverable(self) -> bool:
        # Terminal iff: (a) numeric enabled, (b) energy fluent modeled,
        # (c) all robots have energy < 1, and (d) no charger at current location.
        if not self.enable_numeric or not self._has_energy_support():
            return False
        robots = self._robots()
        if not robots:
            return False
        all_empty = all(self._energy_val(r) < 1.0 for r in robots)
        if not all_empty:
            return False
        return not self._can_recharge_now()

    def _advance_time(self, dur: float, info: Dict[str, Any]):
        """Advance the internal episode clock; domains/tasks never touch time."""
        self.time += float(dur)
        if self.enable_durations and self.time_limit is not None and self.time > self.time_limit:
            # Mark terminal right away
            self.done = True
            info["outcome"] = "timeout"
            info["reason"] = "time_limit_exceeded"

    def _post_apply_success(self, info: Dict[str, Any]):
        reward = self.step_penalty

        # Already terminal (e.g., timeout flagged in _advance_time)
        if self.done:
            pass
        else:
            # Rule-based termination (success/failure/alt endings)
            for r in self.termination_rules:
                try:
                    if self._eval_expr(r["when"]):
                        self.done = True
                        info["outcome"] = r.get("outcome", "terminal")
                        info["terminal_rule"] = r.get("name", "term")
                        reward += float(r.get("reward", 0.0))
                        break
                except Exception:
                    continue

            # Energy exhaustion (only if not already terminal)
            if not self.done and self._energy_depleted_unrecoverable():
                self.done = True
                info["outcome"] = "out_of_energy"
                info["reason"] = "energy_depleted"

            # Step cap → timeout (only if not already terminal)
            elif not self.done and self.steps >= self.max_steps:
                self.done = True
                info["outcome"] = "timeout"
                info["reason"] = "max_steps_exceeded"

        if not self.done and "outcome" not in info:
            info["outcome"] = "ongoing"

        # snapshot last-turn messages BEFORE obs clears the per-step buffer
        last_turn = self._step_messages[:]
        obs = self._obs()
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

        # Exhausted when count > allowed retries.
        # Example: retries=0 → first invalid (count=1) is terminal.
        exhausted = (self._retries_used_this_turn > self.illegal_move_retries)
        if exhausted:
            self.done = True
            return f"Invalid: {msg}\nGame Over.", self.invalid_penalty, True, info

        # Non-terminal retry: same observation with a warning; do NOT advance world/time/steps.
        retries_left = max(self.illegal_move_retries - self._retries_used_this_turn + 1, 0)
        retry_hdr = (
            f"Invalid: {msg}\n"
            f"Please try again. Retries left this turn: {retries_left}\n\n"
        )
        return retry_hdr + self._obs(), self.invalid_retry_penalty, False, info

    def _obs(self) -> str:
        # affordances (on/off decided by formatter)
        aff = self.formatter.generate_affordances(
            self.world, self.static_facts, enable_numeric=self.enable_numeric
        )

        # Show ONLY last-turn messages in the obs, then clear the per-step buffer
        step_msgs = self._step_messages[:]
        text = self.formatter.format_obs(
            world=self.world,
            steps=self.steps,
            max_steps=self.max_steps,
            time_val=self.time,
            durations_on=self.enable_durations,
            messages=step_msgs,
            affordances=aff,
            time_limit=self.time_limit,
            termination_rules=self.termination_rules,
        )
        self._step_messages.clear()
        return text


    def _push_msg(self, s: str):
        """Append a message to the last-turn buffer and the full log."""
        self._step_messages.append(s)
        self.messages.append(s)

    def _eval_expr(self, s: str) -> bool:
        # uses existing clause evaluator (supports numeric & boolean)
        try:
            return eval_clause(self.world, self.static_facts, s, {}, enable_numeric=self.enable_numeric)
        except Exception:
            return False

    def _rs_phi(self, world: WorldState) -> float:
        return sum(w for (expr, w, _) in self.rs_milestones if self._eval_expr(expr))

    def _energy_cap(self, r: str) -> float | None:
        if not self.enable_numeric:
            return None
        cap = self.world.get_fluent("battery-cap", (r,))
        return cap if (cap > 0.0 or ("battery-cap", (r,)) in self.world.fluents) else None

    def _enforce_energy_bounds(self):
        """Clamp energy to [0, battery-cap] for all robots (if modeled)."""
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

    # ---------- Invalid report rendering ----------
    def _format_invalid_report(self, act_name: str, args: Tuple[str, ...], failures: List[EvalNode]) -> str:
        """Pretty, grounded, NL+PDDL explanation of *why* a move is invalid."""
        # Best-effort NL mapper (predicates only)
        from ..pddl.nl_mapper import NLMapper
        nl = NLMapper(self.domain)

        def _pred_nl(grounded: str) -> Optional[str]:
            # try to NL-ify "(pred a b)" when it's a known predicate
            try:
                from ..pddl.grounding import parse_grounded
                name, gargs = parse_grounded(grounded)
                if name in self.domain.predicates:
                    return nl.pred_to_text((name, gargs))
            except Exception:
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
                # show child and say it's true
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
                # literal
                g = n.grounded or n.expr
                nl_line = _pred_nl(g)
                lines.append(f"{bullet}Required: {g}" + (f" — \"{nl_line}\"" if nl_line else ""))
                if n.satisfied:
                    lines.append(f"{pad}    Actually: OK.")
                else:
                    # try to provide a hint for some common patterns
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
