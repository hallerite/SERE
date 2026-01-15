from typing import Any, Dict, Tuple, List, Optional
from sere.pddl.domain_spec import ActionSpec, Predicate
from sere.pddl.grounding import ground_literal
from sere.core.semantics import apply_num_eff, eval_clause, trace_clause, EvalNode, _iter_quantifier_bindings
from sere.core.world_state import WorldState
from . import rendering

def step_one(env, name: str, args: Tuple[str, ...]):
    info: Dict[str, Any] = {}

    if name not in env.domain.actions:
        return env._illegal(f"Unknown action '{name}'", info)

    act: ActionSpec = env.domain.actions[name]

    expected, got = len(act.params), len(args)
    if got != expected:
        return env._illegal(f"Arity mismatch for action '{name}': expected {expected}, got {got}", info)

    type_err = _check_arg_types(env, act, args)
    if type_err:
        return env._illegal(type_err, info)

    static_err = _check_static_effects(env, act)
    if static_err:
        return env._illegal(static_err, info)

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
    derived_cache = {}
    for s in (act.pre or []):
        node = trace_clause(
            env.world,
            env.static_facts,
            s,
            bind,
            enable_numeric=env.enable_numeric,
            _derived_cache=derived_cache,
        )
        if not node.satisfied:
            failures.append(node)
    if failures:
        return env._illegal(env._format_invalid_report(name, args, failures), info)

    _, add, dele = ground_inst(env, act, args)

    dele = _expand_delete_patterns(dele, env.world.facts)
    add  = _expand_add_patterns(add, env.world, act, args)

    pre_world: WorldState | None = None
    if env.enable_conditional and act.cond:
        pre_world = WorldState(
            domain=env.world.domain,
            objects=env.world.objects,
            facts=set(env.world.facts),
            fluents=dict(env.world.fluents),
        )

    env.world.apply(add, dele)

    if env.enable_conditional and act.cond:
        derived_cache = {}
        cond_world = pre_world or env.world
        for cb in act.cond:
            for b2 in _iter_conditional_binds(cond_world, cb, bind):
                ok = True
                for w in cb.when:
                    if not eval_clause(
                        cond_world,
                        env.static_facts,
                        w,
                        b2,
                        enable_numeric=env.enable_numeric,
                        derived_cache=derived_cache,
                    ):
                        ok = False
                        break
                if ok:
                    for a in cb.add:
                        _, litp = ground_literal(a, b2)
                        env.world.facts.add(litp)
                    for d in cb.delete:
                        _, litp = ground_literal(d, b2)
                        env.world.facts.discard(litp)
                    if env.enable_numeric:
                        for ne in cb.num_eff:
                            apply_num_eff(env.world, ne, b2, info)
                    for m in cb.messages:
                        rendering.push_msg(env, format_msg(env, m, b2))

    if env.enable_numeric and act.num_eff:
        for ne in act.num_eff:
            apply_num_eff(env.world, ne, bind, info)

    env._enforce_energy_bounds()

    if getattr(act, "outcomes", None):
        derived_cache = {}
        valid = []
        for oc in act.outcomes:
            ok = True
            for w in (oc.when or []):
                if not eval_clause(
                    env.world,
                    env.static_facts,
                    w,
                    bind,
                    enable_numeric=env.enable_numeric,
                    derived_cache=derived_cache,
                ):
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

    errs = env.world.validate_invariants()
    for pl in env.plugins:
        errs += pl.validate(env.world, env.static_facts)
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
            derived_cache = {}
            for i, (expr, w, once) in enumerate(env.rs_milestones):
                if once and i in env._rs_seen:
                    continue
                if env._eval_expr(expr, derived_cache=derived_cache):
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


def step_joint(env, plan: List[Tuple[str, Tuple[str, ...]]]):
    info: Dict[str, Any] = {"joint": True}

    if not plan:
        return env._illegal("Joint step requires at least one action.", info)

    robots = env._robots()
    if not robots:
        return env._illegal("Joint step requires at least one robot.", info)
    robots = sorted(robots)

    actions_by_robot: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    acts: List[Tuple[Optional[ActionSpec], str, Tuple[str, ...]]] = []

    durations: List[float] = []
    for name, args in plan:
        lname = name.lower()
        if lname == "idle":
            if len(args) != 1:
                return env._illegal("Idle expects exactly one robot argument: (idle r).", info)
            r = args[0]
            if r not in robots:
                return env._illegal(f"Unknown robot '{r}' for idle.", info)
            if r in actions_by_robot:
                return env._illegal(f"Multiple actions provided for robot '{r}'.", info)
            actions_by_robot[r] = (lname, args)
            acts.append((None, lname, args))
            durations.append(0.0)
            continue

        if lname not in env.domain.actions:
            return env._illegal(f"Unknown action '{name}'", info)
        act: ActionSpec = env.domain.actions[lname]

        expected, got = len(act.params), len(args)
        if got != expected:
            return env._illegal(
                f"Arity mismatch for action '{name}': expected {expected}, got {got}", info
            )

        type_err = _check_arg_types(env, act, args)
        if type_err:
            return env._illegal(type_err, info)

        static_err = _check_static_effects(env, act)
        if static_err:
            return env._illegal(static_err, info)

        if getattr(act, "duration_var", None):
            n_name = act.duration_var
            bind_tmp = {var: val for (var, _), val in zip(act.params, args)}
            raw = bind_tmp.get(n_name)
            try:
                dur_multiplier = float(raw)
            except Exception:
                return env._illegal(f"Bad duration_var '{n_name}': {raw!r}", info)
            if dur_multiplier <= 0.0:
                return env._illegal(f"Duration multiplier '{n_name}' must be > 0.", info)

        r = _robot_sym_from_params(act, args)
        if not r:
            return env._illegal(f"Action '{name}' must include a robot parameter in joint mode.", info)
        if r in actions_by_robot:
            return env._illegal(f"Multiple actions provided for robot '{r}'.", info)
        actions_by_robot[r] = (lname, args)
        acts.append((act, lname, args))
        if env.enable_durations:
            if isinstance(act.duration_var, str) and isinstance(act.duration_unit, (int, float)):
                sval = args[[i for i, (v, _t) in enumerate(act.params) if v == act.duration_var][0]]
                durations.append(float(act.duration_unit) * float(sval))
            else:
                durations.append(float(act.duration) if (act.duration is not None) else env.default_duration)

    missing = [r for r in robots if r not in actions_by_robot]
    if missing:
        return env._illegal(
            f"Joint step requires one action per robot; missing for {missing}.", info
        )

    derived_cache: Dict[Tuple[str, Tuple[str, ...]], bool] = {}
    failures = []
    for act, name, args in acts:
        if act is None:
            continue
        bind = {var: val for (var, _), val in zip(act.params, args)}
        failed = []
        for s in (act.pre or []):
            node = trace_clause(
                env.world,
                env.static_facts,
                s,
                bind,
                enable_numeric=env.enable_numeric,
                _derived_cache=derived_cache,
            )
            if not node.satisfied:
                failed.append(node)
        if failed:
            failures.append((name, args, failed))

    if failures:
        parts = []
        for name, args, failed in failures:
            parts.append(env._format_invalid_report(name, args, failed))
        return env._illegal("Joint preconditions failed:\n\n" + "\n\n".join(parts), info)

    # Snapshot for conditional/outcome evaluation
    pre_world = WorldState(
        domain=env.world.domain,
        objects=env.world.objects,
        facts=set(env.world.facts),
        fluents=dict(env.world.fluents),
    )

    add_all: List[Predicate] = []
    del_all: List[Predicate] = []
    num_eff_all: List[Tuple[str, Dict[str, str]]] = []
    messages: List[str] = []

    for act, name, args in acts:
        if act is None:
            continue
        bind = {var: val for (var, _), val in zip(act.params, args)}

        _, add, dele = ground_inst(env, act, args)
        dele = _expand_delete_patterns(dele, pre_world.facts)
        add = _expand_add_patterns(add, pre_world, act, args)
        add_all.extend(add)
        del_all.extend(dele)

        if env.enable_conditional and act.cond:
            derived_cache = {}
            for cb in act.cond:
                for b2 in _iter_conditional_binds(pre_world, cb, bind):
                    ok = True
                    for w in cb.when:
                        if not eval_clause(
                            pre_world,
                            env.static_facts,
                            w,
                            b2,
                            enable_numeric=env.enable_numeric,
                            derived_cache=derived_cache,
                        ):
                            ok = False
                            break
                    if ok:
                        for a in cb.add:
                            _, litp = ground_literal(a, b2)
                            add_all.append(litp)
                        for d in cb.delete:
                            _, litp = ground_literal(d, b2)
                            del_all.append(litp)
                        if env.enable_numeric:
                            for ne in cb.num_eff:
                                num_eff_all.append((ne, b2))
                        for m in cb.messages:
                            messages.append(format_msg(env, m, b2))

        if env.enable_numeric and act.num_eff:
            for ne in act.num_eff:
                num_eff_all.append((ne, bind))

    add_set = set(add_all)
    del_set = set(del_all)
    conflicts = add_set & del_set
    if conflicts:
        return env._illegal(
            f"Joint effects conflict on {sorted(conflicts)}.", info
        )

    env.world.apply(add_all, del_all)

    if env.enable_numeric:
        for expr, bind in num_eff_all:
            apply_num_eff(env.world, expr, bind, info)

    env._enforce_energy_bounds()

    # Outcomes evaluated on post-base state (shared snapshot)
    outcome_add: List[Predicate] = []
    outcome_del: List[Predicate] = []
    outcome_num: List[Tuple[str, Dict[str, str]]] = []
    outcome_msgs: List[str] = []
    action_statuses: Dict[str, str] = {}
    terminal_outcomes: Set[str] = set()

    for act, name, args in acts:
        if act is None or not getattr(act, "outcomes", None):
            continue
        bind = {var: val for (var, _), val in zip(act.params, args)}
        valid = []
        derived_cache = {}
        for oc in act.outcomes:
            ok = True
            for w in (oc.when or []):
                if not eval_clause(
                    env.world,
                    env.static_facts,
                    w,
                    bind,
                    enable_numeric=env.enable_numeric,
                    derived_cache=derived_cache,
                ):
                    ok = False
                    break
            if ok:
                valid.append(oc)

        choice = None
        roll = None
        totp = None
        if valid:
            if env.enable_stochastic:
                totp = sum(max(0.0, float(getattr(oc, "p", 0.0))) for oc in valid)
                acc = 0.0
                roll = env.rng.random() * totp if (totp and totp > 0.0) else None
                for oc in valid:
                    acc += max(0.0, float(getattr(oc, "p", 0.0)))
                    if roll is not None and roll <= acc:
                        choice = oc
                        break
                if choice is None:
                    choice = valid[-1]
            else:
                def _getp(o):
                    try:
                        return float(getattr(o, "p", 1.0))
                    except Exception:
                        return 1.0
                choice = max(valid, key=_getp)

        if choice:
            add2: List[Predicate] = []
            del2: List[Predicate] = []
            for s in (choice.add or []):
                _, litp = ground_literal(s, bind)
                add2.append(litp)
            for s in (choice.delete or []):
                _, litp = ground_literal(s, bind)
                del2.append(litp)
            del2 = _expand_delete_patterns(del2, env.world.facts)
            add2 = _expand_add_patterns(add2, env.world, act, args)
            outcome_add.extend(add2)
            outcome_del.extend(del2)
            if env.enable_numeric and getattr(choice, "num_eff", None):
                for ne in (choice.num_eff or []):
                    outcome_num.append((ne, bind))
            for m in (choice.messages or []):
                outcome_msgs.append(format_msg(env, m, bind))
            status = getattr(choice, "status", None)
            if status:
                action_statuses[f"{name}{args}"] = str(status)

            if bool(getattr(choice, "terminal", False)):
                terminal_outcomes.add(str(getattr(choice, "episode_outcome", "failed")))
            if env.enable_stochastic:
                info.setdefault("stochastic_outcomes", []).append(
                    {"action": name, "args": list(args), "outcome": getattr(choice, "name", "chosen"),
                     "roll": roll, "total_p": totp}
                )

    outcome_conflicts = set(outcome_add) & set(outcome_del)
    if outcome_conflicts:
        return env._illegal(
            f"Joint outcome effects conflict on {sorted(outcome_conflicts)}.", info
        )

    env.world.apply(outcome_add, outcome_del)
    if env.enable_numeric:
        for expr, bind in outcome_num:
            apply_num_eff(env.world, expr, bind, info)

    env._enforce_energy_bounds()

    for msg in messages:
        rendering.push_msg(env, msg)
    for act, _name, args in acts:
        if act is None:
            continue
        bind = {var: val for (var, _), val in zip(act.params, args)}
        for msg in getattr(act, "messages", []) or []:
            rendering.push_msg(env, format_msg(env, msg, bind))
    for msg in outcome_msgs:
        rendering.push_msg(env, msg)

    if terminal_outcomes:
        if len(terminal_outcomes) > 1:
            return env._illegal(
                f"Conflicting terminal outcomes in joint step: {sorted(terminal_outcomes)}.",
                info,
            )
        env.done = True
        info["outcome"] = next(iter(terminal_outcomes))
        info["outcome_set_by_action"] = True

    if env.enable_durations and durations:
        env._advance_time(max(durations), info)

    errs = env.world.validate_invariants()
    for pl in env.plugins:
        errs += pl.validate(env.world, env.static_facts)
    if errs:
        return env._illegal(f"Postcondition violated: {errs}", info)

    env.steps += 1
    obs, base_r, done, info = env._post_apply_success(info)

    rs_bonus = 0.0
    triggered = []
    if env.rs_milestones:
        if env.rs_mode == "potential":
            phi_now = env._rs_phi(env.world)
            rs_bonus = env.rs_gamma * phi_now - env._rs_phi_prev
            env._rs_phi_prev = phi_now
            if abs(rs_bonus) > 0:
                triggered.append(("potential", rs_bonus))
        else:
            derived_cache = {}
            for i, (expr, w, once) in enumerate(env.rs_milestones):
                if once and i in env._rs_seen:
                    continue
                if env._eval_expr(expr, derived_cache=derived_cache):
                    rs_bonus += w
                    triggered.append((i, expr, w, once))
                    if once:
                        env._rs_seen.add(i)

    env._dbg(f"[JOINT] #{env.steps} acts={len(plan)} "
             f"base={base_r:.2f} shape={rs_bonus:.2f} total={base_r + rs_bonus:.2f} "
             f"outcome={info.get('outcome')} triggered={triggered}")

    if abs(rs_bonus) > 0:
        info["shaping_bonus"] = rs_bonus
    if action_statuses:
        info["action_statuses"] = action_statuses

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
                truth = eval_clause(env.world, env.static_facts, token, {}, enable_numeric=env.enable_numeric)
                s = s.replace(token, "true" if truth else "false")
        except Exception:
            pass
    return s

def _is_number_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False

def _candidates_of_type(env, typ: str) -> List[str]:
    return sorted(
        sym for sym, tys in env.world.objects.items()
        if tys and any(env.domain.is_subtype(t, typ) for t in tys)
    )

def _static_predicates(env) -> set[str]:
    return {
        name for name, spec in (env.domain.predicates or {}).items()
        if getattr(spec, "static", False)
    }

def _effect_pred_name(effect: str) -> Optional[str]:
    try:
        _, litp = ground_literal(effect, {})
    except Exception:
        return None
    return litp[0]

def _check_static_effects(env, act: ActionSpec) -> Optional[str]:
    static_preds = _static_predicates(env)
    if not static_preds:
        return None
    offenders: List[str] = []

    def _scan(effects: List[str], where: str) -> None:
        for eff in effects or []:
            pname = _effect_pred_name(eff)
            if pname and pname in static_preds:
                offenders.append(f"{where}: {eff}")

    _scan(getattr(act, "add", []) or [], "add")
    _scan(getattr(act, "delete", []) or [], "delete")
    for i, cb in enumerate(getattr(act, "cond", []) or []):
        _scan(cb.add, f"cond[{i}].add")
        _scan(cb.delete, f"cond[{i}].delete")
    for i, oc in enumerate(getattr(act, "outcomes", []) or []):
        _scan(oc.add, f"outcomes[{i}].add")
        _scan(oc.delete, f"outcomes[{i}].delete")

    if offenders:
        return (
            f"Action '{act.name}' modifies static predicates, which is disallowed: "
            + "; ".join(offenders)
        )
    return None

def _check_arg_types(env, act: ActionSpec, args: Tuple[str, ...]) -> Optional[str]:
    for (var, typ), arg in zip(act.params, args):
        ptyp = str(typ).lower()
        if ptyp == "number":
            if _is_number_token(arg):
                continue
            return f"Bad argument for '{act.name}': {var} expects a number, got '{arg}'."
        if ptyp not in env.domain.types:
            return f"Bad parameter type for '{act.name}': {var} has unknown type '{ptyp}'."
        obj_types = env.world.objects.get(arg)
        if not obj_types:
            return f"Unknown object '{arg}' for '{act.name}' parameter '{var}'."
        if not any(env.domain.is_subtype(t, ptyp) for t in obj_types):
            choices = _candidates_of_type(env, ptyp)
            hint = f" Candidates: {choices}." if choices else ""
            return (
                f"Type mismatch for '{act.name}': {var} expects {ptyp}, "
                f"got '{arg}' with types {sorted(obj_types)}.{hint}"
            )
    return None

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


def _iter_conditional_binds(cond_world: WorldState, cb, base_bind: Dict[str, str]):
    if getattr(cb, "forall", None):
        for qb in _iter_quantifier_bindings(cond_world, cb.forall):
            merged = dict(base_bind)
            merged.update(qb)
            yield merged
    else:
        yield base_bind

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
