from typing import Any, Dict, Tuple, List, Optional
from sere.pddl.domain_spec import ActionSpec, Predicate
from sere.pddl.grounding import ground_literal
from sere.core.semantics import apply_num_eff, eval_clause, trace_clause, EvalNode
from . import rendering

def step_one(env, name: str, args: Tuple[str, ...]):
    info: Dict[str, Any] = {}

    if name not in env.domain.actions:
        return env._illegal(f"Unknown action '{name}'", info)

    act: ActionSpec = env.domain.actions[name]

    expected, got = len(act.params), len(args)
    if got != expected:
        return env._illegal(f"Arity mismatch for action '{name}': expected {expected}, got {got}", info)
    bind = {var: val for (var, _), val in zip(act.params, args)}

    if getattr(act, "duration_var", None):
        n_name = act.duration_var
        raw = bind.get(n_name)
        try:
            dur_multiplier = float(raw)
        except Exception:
            return env._illegal(f"Bad duration_var '{n_name}': {raw!r}", info)
        if dur_multiplier <= 0.0:
            return env._illegal(f"Duration multiplier '{n_name}' must be > 0.", info)

    failures: List[EvalNode] = []
    for s in (act.pre or []):
        node = trace_clause(env.world, env.static_facts, s, bind, enable_numeric=env.enable_numeric)
        if not node.satisfied:
            failures.append(node)
    if failures:
        return env._illegal(env._format_invalid_report(name, args, failures), info)

    _, add, dele = ground_inst(env, act, args)

    dele = _expand_delete_patterns(dele, env.world.facts)
    add  = _expand_add_patterns(add, env.world, act, args)

    env.world.apply(add, dele)

    if env.enable_conditional and act.cond:
        for cb in act.cond:
            ok = True
            for w in cb.when:
                if not eval_clause(env.world, env.static_facts, w, bind, enable_numeric=env.enable_numeric):
                    ok = False
                    break
            if ok:
                for a in cb.add:
                    _, litp = ground_literal(a, bind)
                    env.world.facts.add(litp)
                for d in cb.delete:
                    _, litp = ground_literal(d, bind)
                    env.world.facts.discard(litp)
                if env.enable_numeric:
                    for ne in cb.num_eff:
                        apply_num_eff(env.world, ne, bind, info)
                for m in cb.messages:
                    rendering.push_msg(env, format_msg(env, m, bind))

    if env.enable_numeric and act.num_eff:
        for ne in act.num_eff:
            apply_num_eff(env.world, ne, bind, info)

    env._enforce_energy_bounds()

    if getattr(act, "outcomes", None):
        valid = []
        for oc in act.outcomes:
            ok = True
            for w in (oc.when or []):
                if not eval_clause(env.world, env.static_facts, w, bind, enable_numeric=env.enable_numeric):
                    ok = False
                    break
            if ok:
                valid.append(oc)

        choice = None
        roll = None
        totp = None

        if env.enable_stochastic:
            totp = sum(max(0.0, float(getattr(oc, "p", 0.0))) for oc in valid)
            if valid and (totp is None or totp <= 0.0):
                info["stochastic_warning"] = "nonpositive_total_probability"
            acc = 0.0
            roll = env.rng.random() * totp if (totp and totp > 0.0) else None
            for oc in valid:
                acc += max(0.0, float(getattr(oc, "p", 0.0)))
                if roll is not None and roll <= acc:
                    choice = oc
                    break
            if choice is None and valid:
                choice = valid[-1]
        else:
            if valid:
                def _getp(o):
                    try:
                        return float(getattr(o, "p", 1.0))
                    except Exception:
                        return 1.0
                choice = max(valid, key=_getp)

        if choice:
            add2 = []
            del2 = []
            for s in (choice.add or []):
                _, litp = ground_literal(s, bind)
                add2.append(litp)
            for s in (choice.delete or []):
                _, litp = ground_literal(s, bind)
                del2.append(litp)
            del2 = _expand_delete_patterns(del2, env.world.facts)
            add2 = _expand_add_patterns(add2, env.world, act, args)
            env.world.apply(add2, del2)

            if env.enable_numeric and getattr(choice, "num_eff", None):
                for ne in (choice.num_eff or []):
                    apply_num_eff(env.world, ne, bind, info)

            for m in (choice.messages or []):
                rendering.push_msg(env, format_msg(env, m, bind))

            info["outcome_branch"] = str(getattr(choice, "name", "chosen"))
            status = getattr(choice, "status", None)
            if status:
                info["action_status"] = str(status)

            if bool(getattr(choice, "terminal", False)):
                env.done = True
                info["outcome"] = str(getattr(choice, "episode_outcome", "failed"))
                info["outcome_set_by_action"] = True

            if env.enable_stochastic:
                info["stochastic_outcome"] = info.get("outcome_branch")
                info["stochastic_roll"] = roll
                info["stochastic_total_p"] = totp

    env._enforce_energy_bounds()

    for msg in getattr(act, "messages", []) or []:
        rendering.push_msg(env, format_msg(env, msg, bind))

    if env.enable_durations:
        vname = getattr(act, "duration_var", None)
        unit = getattr(act, "duration_unit", None)

        if isinstance(vname, str) and isinstance(unit, (int, float)):
            sval = bind.get(vname)
            if sval is None:
                return env._illegal(f"Bad duration var '{vname}' value: {sval!r}", info)
            try:
                dur = float(unit) * float(sval)
            except (TypeError, ValueError):
                return env._illegal(f"Bad duration var '{vname}' value: {sval!r}", info)
        else:
            dur = float(act.duration) if (act.duration is not None) else env.default_duration

        env._advance_time(dur, info)

    for pl in env.plugins:
        errs = pl.validate(env.world, env.static_facts)
        if errs:
            return env._illegal(f"Postcondition violated: {errs}", info)

    env.steps += 1
    obs, base_r, done, info = env._post_apply_success(info)

    rs_bonus = 0.0
    triggered = []  # <-- collect which milestones fired this step

    if env.rs_milestones:
        if env.rs_mode == "potential":
            phi_now = env._rs_phi(env.world)
            rs_bonus = env.rs_gamma * phi_now - env._rs_phi_prev
            env._rs_phi_prev = phi_now
            if abs(rs_bonus) > 0:
                triggered.append(("potential", rs_bonus))
        else:
            for i, (expr, w, once) in enumerate(env.rs_milestones):
                if once and i in env._rs_seen:
                    continue
                if env._eval_expr(expr):
                    rs_bonus += w
                    triggered.append((i, expr, w, once))
                    if once:
                        env._rs_seen.add(i)
                        
    # ---- DEBUG LINE: per-step reward + shaping ----
    env._dbg(f"[STEP] #{env.steps+1} act=({name} {' '.join(args)}) "
             f"base={base_r:.2f} shape={rs_bonus:.2f} total={base_r + rs_bonus:.2f} "
             f"outcome={info.get('outcome')} triggered={triggered}")

    if abs(rs_bonus) > 0:
        info["shaping_bonus"] = rs_bonus

    env._retries_used_this_turn = 0

    return obs, base_r + rs_bonus, done, info


# helpers local to the executor

def ground_inst(env, act: ActionSpec, args: Tuple[str, ...]):
    # indirection to keep engine self-contained
    from sere.pddl.grounding import instantiate
    return instantiate(env.domain, act, args)

def format_msg(env, template: str, bind: Dict[str, str]) -> str:
    s = template
    for k, v in bind.items():
        s = s.replace("{" + k + "}", v)
    import re as _re
    for k, v in bind.items():
        s = _re.sub(rf'\?{_re.escape(k)}(\b|[^A-Za-z0-9_?-])', lambda m: v + (m.group(1) or ''), s)
    for token in _re.findall(r"\([^)]+\)", s):
        try:
            from sere.pddl.grounding import parse_grounded
            name, args = parse_grounded(token)
            if name in ["<", ">", "<=", ">=", "="]:
                continue
            val = env.world.get_fluent(name, args)
            if val != 0.0 or (name, args) in env.world.fluents:
                s = s.replace(token, f"{val:.2f}")
            else:
                truth = env.world.holds((name, args))
                s = s.replace(token, "true" if truth else "false")
        except Exception:
            pass
    return s

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

def _unique_robot_loc(world, r: Optional[str]) -> Optional[str]:
    if not r:
        return None
    locs = [a[1] for (pred, a) in world.facts if pred == "at" and len(a) == 2 and a[0] == r]
    return locs[0] if len(locs) == 1 else None

def _expand_add_patterns(adds: List[Predicate], world, act: ActionSpec, arg_tuple: tuple) -> List[Predicate]:
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
