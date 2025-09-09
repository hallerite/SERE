from typing import Any, Dict, List, Optional, Tuple
import re, random
from ..pddl.domain_spec import DomainSpec, ActionSpec, Predicate
from ..pddl.grounding import parse_grounded, instantiate, ground_literal
from .prompt_formatter import PromptFormatter, PromptFormatterConfig
from .world_state import WorldState
from .invariants import InvariantPlugin

# --- numeric guards/effects ---
NUM_CMP = re.compile(
    r"^\(\s*(<=|>=|<|>|=)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+([+-]?\d+(?:\.\d+)?)\s*\)$"
)
NUM_EFF = re.compile(
    r"^\(\s*(increase|decrease|assign|cost)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+([+-]?\d+(?:\.\d+)?)\s*\)$"
)

def _bind_args(argstr: Optional[str], bind: Dict[str, str]) -> Tuple[str, ...]:
    toks = argstr.split() if argstr else []
    return tuple(bind.get(t[1:], t) if t.startswith("?") else t for t in toks)

def _eval_num_pre(world: WorldState, expr: str, bind: Dict[str,str]) -> bool:
    m = NUM_CMP.match(expr.strip())
    if not m: raise ValueError(f"Bad num_pre: {expr}")
    op, fname, argstr, rhs = m.groups()
    args = _bind_args(argstr, bind)
    val = world.get_fluent(fname, args)
    rhs = float(rhs)
    if op == "<":  return val < rhs
    if op == "<=": return val <= rhs
    if op == ">":  return val > rhs
    if op == ">=": return val >= rhs
    return abs(val - rhs) < 1e-9  # "="

def _apply_num_eff(world: WorldState, expr: str, bind: Dict[str,str], info: Dict[str,Any]):
    m = NUM_EFF.match(expr.strip())
    if not m: raise ValueError(f"Bad num_eff: {expr}")
    op, fname, argstr, delta = m.groups()
    args  = _bind_args(argstr, bind)
    delta = float(delta)
    if op == "assign":
        world.set_fluent(fname, args, delta)
    elif op == "increase":
        world.set_fluent(fname, args, world.get_fluent(fname, args) + delta)
    elif op == "decrease":
        world.set_fluent(fname, args, world.get_fluent(fname, args) - delta)
    elif op == "cost":
        info["action_cost"] = info.get("action_cost", 0.0) + delta

def extract_tag(s: str, tag: str) -> Optional[str]:
    start, end = f"<{tag}>", f"</{tag}>"
    i, j = s.find(start), s.find(end)
    return None if i < 0 or j < 0 else s[i+len(start):j].strip()

def _format_msg(template: str, bind: Dict[str, str], world: WorldState) -> str:
    """
    Replace ?vars with their bound symbols.
    Also supports simple inline probes:
      - "(quality ?a)" → current numeric fluent value
      - "(pred ?x ...)" → truthy/falsey probe prints "true"/"false"
    We keep it intentionally lightweight to avoid dragging the parser in.
    """
    s = template
    for k, v in bind.items():
        s = re.sub(rf'\?{re.escape(k)}(\b|[^A-Za-z0-9_?-])', lambda m: v + (m.group(1) or ''), s)

    # inline probes of form "(name arg1 arg2)"
    for token in re.findall(r"\([^)]+\)", s):
        try:
            name, args = parse_grounded(token)
            if name in ["<", ">", "<=", ">=", "="]:
                continue
            # try fluent read
            val = world.get_fluent(name, args)
            # if fluent missing and args non-empty, maybe it's a predicate probe
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
                 static_facts: set, goal: List[tuple],
                 formatter_config: Optional[dict] = None,
                 formatter_obj: Optional[PromptFormatter] = None,
                 plugins: Optional[List[InvariantPlugin]] = None,
                 max_steps: int = 40, step_penalty: float = -0.01,
                 invalid_penalty: float = -0.1, goal_reward: float = 1.0,
                 *,
                 enable_numeric: bool = False,
                 enable_conditional: bool = False,
                 enable_durations: bool = False,
                 enable_stochastic: bool = True, 
                 max_messages: int = 8,
                 time_limit: Optional[float] = None,

                 visible_fluents: Optional[List[str]] = None,   # which fluents to print in the obs
                 fluents_precision: int = 2,
                 show_fluent_deltas: bool = True,

                 default_duration: float = 1.0,
                 seed: Optional[int] = None):
        self.domain, self.world = domain, world
        self.static_facts, self.goal = static_facts, goal

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
        self.step_penalty, self.invalid_penalty, self.goal_reward = step_penalty, invalid_penalty, goal_reward
        self.steps = 0
        self.done = False
        self.enable_numeric = enable_numeric
        self.enable_conditional = enable_conditional
        self.enable_durations = enable_durations
        self.enable_stochastic = enable_stochastic

        self.default_duration = float(default_duration)
        self.time_limit = time_limit
        self.time = 0.0
        
        self.visible_fluents = visible_fluents or ["*"]   # glob-style
        self.fluents_precision = int(fluents_precision)
        self.show_fluent_deltas = bool(show_fluent_deltas)
        self._prev_fluents: Dict[Tuple[str, Tuple[str,...]], float] = {}

        self.messages: List[str] = []
        self.max_messages = int(max_messages)


        self.rng = random.Random(seed)

    def reset(self, *, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        self.steps, self.done = 0, False
        self.time = 0.0
        errs = self.world.validate_invariants()
        for pl in self.plugins: errs += pl.validate(self.world)
        if errs:
            raise ValueError(f"Invariant errors: {errs}")

        self.messages.clear()
        self._system_prompt_cache = self.formatter.build_system_prompt(
            world=self.world,
            static_facts=self.static_facts,
            goal=self.goal
        )

        obs_text = self._obs()  # PLAIN TEXT ONLY
        info = {
            "system_prompt": self._system_prompt_cache,
            "problem_pddl": self.world.to_problem_pddl("instance", self.static_facts, self.goal),
            "features": dict(numeric=self.enable_numeric,
                             conditional=self.enable_conditional,
                             durations=self.enable_durations,
                             stochastic=self.enable_stochastic)
        }
        return obs_text, info

    def step(self, text: str):
        if self.done: raise RuntimeError("Episode finished. Call reset().")

        self._prev_fluents = dict(self.world.fluents)

        info: Dict[str, Any] = {}
        move = extract_tag(text, "move")
        if not move:
            return self._illegal("Missing <move>(...)</move>.", info)

        try:
            name, args = parse_grounded(move)
        except Exception as e:
            return self._illegal(f"Parse error: {e}", info)

        # Built-in wait
        if name == "wait":
            dur = float(args[0]) if args else 1.0
            if self.enable_durations:
                self._advance_time(dur, info)
            self.steps += 1
            return self._post_apply_success(info)

        if name not in self.domain.actions:
            return self._illegal(f"Unknown action '{name}'", info)

        act: ActionSpec = self.domain.actions[name]
        bind = {var: val for (var,_), val in zip(act.params, args)}

        # Ground positives only
        pre_pos: List[Predicate] = []
        neg_lits: List[Predicate] = []
        for s in act.pre:
            is_neg, litp = ground_literal(s, bind)
            (neg_lits if is_neg else pre_pos).append(litp)

        # Numeric guards
        if self.enable_numeric and act.num_pre:
            for np in act.num_pre:
                if not _eval_num_pre(self.world, np, bind):
                    return self._illegal(f"Numeric precondition failed: {np}", info)

        # Classic preconditions
        missing = self.world.check_preconds(pre_pos, extra=self.static_facts)
        if missing:
            return self._illegal(f"Preconditions not satisfied: {missing}", info)

        # Negative preconditions (route through world.holds so derived preds work)
        for litp in neg_lits:
            if self.world.holds(litp) or (litp in self.static_facts):
                return self._illegal(
                    f"Negative precondition violated: (not ({litp[0]} {' '.join(litp[1])}))",
                    info
                )

        # Apply base effects
        _, add, dele = instantiate(self.domain, act, args)

        # ---------- Engine niceties (lazy, arity-safe) ----------
        def _is_unbound(x: Any) -> bool:
            return isinstance(x, str) and x.startswith("?")

        # Wildcard deletes: unbound vars act as wildcards (delete all matches)
        def _expand_delete_patterns(deletes: List[Predicate], facts: set) -> List[Predicate]:
            expanded: List[Predicate] = []
            for (pred, argtup) in deletes:
                if any(_is_unbound(a) for a in argtup):
                    for (p, a) in list(facts):
                        if p != pred: continue
                        ok = True
                        for i, pat in enumerate(argtup):
                            if _is_unbound(pat): continue
                            if i >= len(a) or pat != a[i]:
                                ok = False; break
                        if ok: expanded.append((p, a))
                else:
                    expanded.append((pred, argtup))
            return expanded

        def _robot_sym_from_params(act: ActionSpec, arg_tuple: tuple) -> Optional[str]:
            for i, (_, ty) in enumerate(act.params):
                if ty.lower() == "robot":
                    return arg_tuple[i] if i < len(arg_tuple) else None
            return None

        # Unique robot location (arity-safe)
        def _unique_robot_loc(world: WorldState, r: Optional[str]) -> Optional[str]:
            if not r: return None
            locs = [a[1] for (pred, a) in world.facts
                    if pred == "at" and len(a) == 2 and a[0] == r]
            return locs[0] if len(locs) == 1 else None


        # Add-at-robot-location: only when we actually see an unbound 'obj-at' add
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

        # Conditional effects
        if self.enable_conditional and act.cond:
            for cb in act.cond:
                ok = True
                for w in cb.when:
                    w = w.strip()
                    if w.startswith("(") and not w.startswith("(not") and any(w.startswith(f"({op}") for op in ["<", ">", "<=", ">=", "="]):
                        if not self.enable_numeric or not _eval_num_pre(self.world, w, bind):
                            ok = False; break
                    else:
                        is_neg, litw = ground_literal(w, bind)
                        h = (litw in self.world.facts) or (litw in self.static_facts)
                        if (not is_neg and not h) or (is_neg and h):
                            ok = False; break
                if ok:
                    for a in cb.add:
                        _, litp = ground_literal(a, bind); self.world.facts.add(litp)
                    for d in cb.delete:
                        _, litp = ground_literal(d, bind); self.world.facts.discard(litp)
                    if self.enable_numeric:
                        for ne in cb.num_eff: _apply_num_eff(self.world, ne, bind, info)

        # Emit base action messages
        for msg in getattr(act, "messages", []) or []:
            self.messages.append(_format_msg(msg, bind, self.world))

        # Emit conditional block messages
        if self.enable_conditional and act.cond:
            for cb in act.cond:
                ok = True
                # (we already computed ok earlier; do it again or refactor—keeping it simple)
                for w in cb.when:
                    w = w.strip()
                    if w.startswith("(") and not w.startswith("(not") and any(w.startswith(f"({op}") for op in ["<", ">", "<=", ">=", "="]):
                        if not self.enable_numeric or not _eval_num_pre(self.world, w, bind):
                            ok = False; break
                    else:
                        is_neg, litw = ground_literal(w, bind)
                        h = (litw in self.world.facts) or (litw in self.static_facts)
                        if (not is_neg and not h) or (is_neg and h):
                            ok = False; break
                if ok:
                    for m in cb.messages:
                        self.messages.append(_format_msg(m, bind, self.world))


        # Numeric effects (unconditional)
        if self.enable_numeric and act.num_eff:
            for ne in act.num_eff:
                _apply_num_eff(self.world, ne, bind, info)
        
        # Stochastic outcomes
        if self.enable_stochastic and getattr(act, "outcomes", None):
            # Filter outcomes by optional 'when' guards
            valid = []
            for oc in act.outcomes:
                ok = True
                for w in oc.when or []:
                    w = w.strip()
                    if w.startswith("(") and not w.startswith("(not") and any(w.startswith(f"({op}") for op in ["<", ">", "<=", ">=", "="]):
                        if not self.enable_numeric or not _eval_num_pre(self.world, w, bind):
                            ok = False; break
                    else:
                        is_neg, litw = ground_literal(w, bind)
                        h = (litw in self.world.facts) or (litw in self.static_facts)
                        if (not is_neg and not h) or (is_neg and h):
                            ok = False; break
                if ok:
                    valid.append(oc)

            # Normalize and sample
            totp = sum(max(0.0, float(oc.p)) for oc in valid)
            if valid and totp <= 0:
                info["stochastic_warning"] = "nonpositive_total_probability"
            choice = None
            roll = self.rng.random() * totp if totp > 0 else None
            acc = 0.0
            for oc in valid:
                acc += max(0.0, float(oc.p))
                if roll is not None and roll <= acc:
                    choice = oc
                    break
            if choice is None and valid:
                # fallback: pick last if totp==0 or num issues
                choice = valid[-1]

            if choice:
                # apply choice effects
                add = []
                dele = []
                for s in choice.add or []:
                    _, litp = ground_literal(s, bind); add.append(litp)
                for s in (choice.delete or []):
                    _, litp = ground_literal(s, bind); dele.append(litp)
                # Use same wildcard-expansion niceties
                dele = _expand_delete_patterns(dele, self.world.facts)
                add  = _expand_add_patterns(add, self.world, act, args)
                self.world.apply(add, dele)

                # numeric effects
                if self.enable_numeric and choice.num_eff:
                    for ne in choice.num_eff:
                        _apply_num_eff(self.world, ne, bind, info)

                # messages
                for m in choice.messages or []:
                    self.messages.append(_format_msg(m, bind, self.world))

                info["stochastic_outcome"] = getattr(choice, "name", "chosen")
                info["stochastic_roll"] = roll
                info["stochastic_total_p"] = totp


        # Time / duration
        if self.enable_durations:
            dur = float(act.duration) if (act.duration is not None) else self.default_duration
            self._advance_time(dur, info)

        # Plugins (post)
        for pl in self.plugins:
            errs = pl.validate(self.world)
            if errs:
                return self._illegal(f"Postcondition violated: {errs}", info)

        # Step bookkeeping and terminal check
        self.steps += 1
        return self._post_apply_success(info)

    # --- helpers ---
    def _advance_time(self, dur: float, info: Dict[str, Any]):
        self.time += dur
        # keep a matching fluent if user defined (:functions elapsed)
        try:
            self.world.set_fluent("elapsed", tuple(), self.time)
        except Exception:
            pass
        if self.enable_durations and self.time_limit is not None and self.time > self.time_limit:
            # Mark terminal right away
            self.done = True
            info["outcome"] = "loss"
            info["reason"] = "time_limit_exceeded"

    def _post_apply_success(self, info: Dict[str, Any]):
        reward = self.step_penalty
        if all(self.world.holds(g) for g in self.goal):
            self.done = True
            reward += self.goal_reward
            info["outcome"] = "win"
        elif self.done:  # e.g., time limit above
            pass
        elif self.steps + 1 > self.max_steps:
            self.done = True
            info["outcome"] = "loss"
        else:
            info["outcome"] = "ongoing"
        info["messages"] = self.messages[-self.max_messages:]
        return self._obs(), reward, self.done, info
    
    def _illegal(self, msg, info):
        info.update({
            "invalid_move": True,
            "error": msg,
            "outcome": "invalid",
            "messages": self.messages[-self.max_messages:],  # add
        })
        self.done = True
        return f"Invalid: {msg}\nGame Over.", self.invalid_penalty, True, info

    def _fluent_visible(self, name: str) -> bool:
        import fnmatch
        return any(fnmatch.fnmatch(name, pat) for pat in self.visible_fluents)

    def _obs(self) -> str:
        # affordances (on/off decided by formatter)
        aff = self.formatter.generate_affordances(self.world, self.static_facts)

        return self.formatter.format_obs(
            world=self.world,
            static_facts=self.static_facts,
            goal=self.goal,
            steps=self.steps,
            max_steps=self.max_steps,
            time_val=self.time,
            durations_on=self.enable_durations,
            messages=self.messages[-self.max_messages:],
            prev_fluents=self._prev_fluents,
            affordances=aff,
        )