from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional
from sere.pddl.domain_spec import ActionSpec, Predicate
from sere.pddl.grounding import ground_literal
from sere.core.semantics import apply_num_eff, eval_clause, eval_trace, _iter_quantifier_bindings
from sere.core.world_state import WorldState
from . import rendering


@dataclass
class ActionCall:
    act: Optional[ActionSpec]
    name: str
    args: Tuple[str, ...]
    bind: Optional[Dict[str, str]] = None

def step_one(env, name: str, args: Tuple[str, ...]):
    info: Dict[str, Any] = {}

    act, bind, err = _resolve_action(env, name, args)
    if err:
        return env._illegal(err, info)

    durations = None
    if env.enable_durations:
        dur, err = _duration_for_action(env, act, bind)
        if err:
            return env._illegal(err, info)
        durations = [dur]

    call = ActionCall(act=act, name=name, args=args, bind=bind)
    return _execute_actions(
        env,
        [call],
        info,
        joint=False,
        durations=durations,
        warn_outcome_probs=True,
    )


def step_joint(env, plan: List[Tuple[str, Tuple[str, ...]]]):
    info: Dict[str, Any] = {"joint": True}

    if not plan:
        return env._illegal("Joint step requires at least one action.", info)

    robots = env._robots()
    if not robots:
        return env._illegal("Joint step requires at least one robot.", info)
    robots = sorted(robots)

    calls, durations, err = _collect_joint_calls(env, plan, robots)
    if err:
        return env._illegal(err, info)

    return _execute_actions(
        env,
        calls,
        info,
        joint=True,
        durations=durations if env.enable_durations else None,
    )


def _execute_actions(
    env,
    calls: List[ActionCall],
    info: Dict[str, Any],
    *,
    joint: bool,
    durations: Optional[List[float]] = None,
    warn_outcome_probs: bool = False,
):
    derived_cache: Dict[Tuple[str, Tuple[str, ...]], bool] = {}
    failures = []
    for call in calls:
        if call.act is None:
            continue
        bind = call.bind or {var: val for (var, _), val in zip(call.act.params, call.args)}
        call.bind = bind
        failed = []
        for s in (call.act.pre or []):
            node = eval_trace(
                env.world,
                env.static_facts,
                s,
                bind,
                enable_numeric=env.enable_numeric,
                derived_cache=derived_cache,
            )
            if not node.satisfied:
                failed.append(node)
        if failed:
            failures.append((call.name, call.args, failed))

    if failures:
        if joint:
            parts = []
            for name, args, failed in failures:
                parts.append(env._format_invalid_report(name, args, failed))
            return env._illegal("Joint preconditions failed:\n\n" + "\n\n".join(parts), info)
        name, args, failed = failures[0]
        return env._illegal(env._format_invalid_report(name, args, failed), info)

    needs_snapshot = joint or (
        env.enable_conditional
        and any(call.act is not None and call.act.cond for call in calls)
    )
    pre_world: WorldState | None = None
    if needs_snapshot:
        pre_world = WorldState(
            domain=env.world.domain,
            objects=env.world.objects,
            facts=set(env.world.facts),
            fluents=dict(env.world.fluents),
        )

    base_delta = ActionDelta()
    cond_delta = ActionDelta()
    action_num_eff: List[Tuple[str, Dict[str, str]]] = []
    cond_messages: List[str] = []

    for call in calls:
        if call.act is None:
            continue
        bind = call.bind or {var: val for (var, _), val in zip(call.act.params, call.args)}
        call.bind = bind

        act_base = _compute_base_delta(env, call.act, call.args, world=pre_world)
        base_delta.add.extend(act_base.add)
        base_delta.delete.extend(act_base.delete)

        if env.enable_conditional and call.act.cond:
            act_cond = _compute_conditional_delta(env, call.act, bind, pre_world or env.world)
            cond_delta.add.extend(act_cond.add)
            cond_delta.delete.extend(act_cond.delete)
            cond_delta.num_eff.extend(act_cond.num_eff)
            if joint:
                for template, b2 in act_cond.messages:
                    cond_messages.append(format_msg(env, template, b2))
            else:
                cond_delta.messages.extend(act_cond.messages)

        if env.enable_numeric and call.act.num_eff:
            for ne in call.act.num_eff:
                action_num_eff.append((ne, bind))

    if joint:
        add_set = set(base_delta.add) | set(cond_delta.add)
        del_set = set(base_delta.delete) | set(cond_delta.delete)
        conflicts = add_set & del_set
        if conflicts:
            return env._illegal(
                f"Joint effects conflict on {sorted(conflicts)}.", info
            )

    _apply_delta(env, base_delta, info, push_messages=not joint)
    if env.enable_conditional and (
        cond_delta.add or cond_delta.delete or cond_delta.num_eff or cond_delta.messages
    ):
        _apply_delta(env, cond_delta, info, push_messages=not joint)

    if env.enable_numeric:
        for expr, bind in action_num_eff:
            apply_num_eff(env.world, expr, bind, info)

    env._enforce_energy_bounds()

    outcome_add: List[Predicate] = []
    outcome_del: List[Predicate] = []
    outcome_num: List[Tuple[str, Dict[str, str]]] = []
    outcome_msgs: List[str] = []
    outcome_msg_templates: List[Tuple[str, Dict[str, str]]] = []
    action_statuses: Dict[str, str] = {}
    terminal_outcomes: set[str] = set()

    for call in calls:
        if call.act is None or not getattr(call.act, "outcomes", None):
            continue
        bind = call.bind or {var: val for (var, _), val in zip(call.act.params, call.args)}
        call.bind = bind
        choice, roll, totp = _select_outcome(
            env,
            call.act,
            bind,
            warn_nonpositive_total_prob=warn_outcome_probs,
            info=info if warn_outcome_probs else None,
        )
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
            add2 = _expand_add_patterns(add2, env.world, call.act, call.args)
            outcome_add.extend(add2)
            outcome_del.extend(del2)
            if env.enable_numeric and getattr(choice, "num_eff", None):
                for ne in (choice.num_eff or []):
                    outcome_num.append((ne, bind))
            for m in (choice.messages or []):
                if joint:
                    outcome_msgs.append(format_msg(env, m, bind))
                else:
                    outcome_msg_templates.append((m, bind))

            if joint:
                status = getattr(choice, "status", None)
                if status:
                    action_statuses[f"{call.name}{call.args}"] = str(status)
            else:
                info["outcome_branch"] = str(getattr(choice, "name", "chosen"))
                status = getattr(choice, "status", None)
                if status:
                    info["action_status"] = str(status)

            if bool(getattr(choice, "terminal", False)):
                terminal_outcomes.add(str(getattr(choice, "episode_outcome", "failed")))

            if env.enable_stochastic:
                if joint:
                    info.setdefault("stochastic_outcomes", []).append(
                        {
                            "action": call.name,
                            "args": list(call.args),
                            "outcome": getattr(choice, "name", "chosen"),
                            "roll": roll,
                            "total_p": totp,
                        }
                    )
                else:
                    info["stochastic_outcome"] = info.get("outcome_branch")
                    info["stochastic_roll"] = roll
                    info["stochastic_total_p"] = totp

    if joint:
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

    if not joint and outcome_msg_templates:
        for template, bind in outcome_msg_templates:
            outcome_msgs.append(format_msg(env, template, bind))

    if joint:
        for msg in cond_messages:
            rendering.push_msg(env, msg)

    if joint:
        for call in calls:
            if call.act is None:
                continue
            bind = call.bind or {var: val for (var, _), val in zip(call.act.params, call.args)}
            call.bind = bind
            for msg in getattr(call.act, "messages", []) or []:
                rendering.push_msg(env, format_msg(env, msg, bind))
        for msg in outcome_msgs:
            rendering.push_msg(env, msg)
    else:
        for msg in outcome_msgs:
            rendering.push_msg(env, msg)
        for call in calls:
            if call.act is None:
                continue
            bind = call.bind or {var: val for (var, _), val in zip(call.act.params, call.args)}
            call.bind = bind
            for msg in getattr(call.act, "messages", []) or []:
                rendering.push_msg(env, format_msg(env, msg, bind))

    if joint:
        if terminal_outcomes:
            if len(terminal_outcomes) > 1:
                return env._illegal(
                    f"Conflicting terminal outcomes in joint step: {sorted(terminal_outcomes)}.",
                    info,
                )
            env.done = True
            info["outcome"] = next(iter(terminal_outcomes))
            info["outcome_set_by_action"] = True
    else:
        if terminal_outcomes:
            env.done = True
            info["outcome"] = next(iter(terminal_outcomes))
            info["outcome_set_by_action"] = True

    if env.enable_durations and durations:
        if joint:
            env._advance_time(max(durations), info)
        else:
            env._advance_time(durations[0], info)

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

    if joint:
        env._dbg(f"[JOINT] #{env.steps} acts={len(calls)} "
                 f"base={base_r:.2f} shape={rs_bonus:.2f} total={base_r + rs_bonus:.2f} "
                 f"outcome={info.get('outcome')} triggered={triggered}")
    else:
        only = calls[0] if calls else None
        act_name = only.name if only else "?"
        act_args = " ".join(only.args) if only else ""
        env._dbg(f"[STEP] #{env.steps+1} act=({act_name} {act_args}) "
                 f"base={base_r:.2f} shape={rs_bonus:.2f} total={base_r + rs_bonus:.2f} "
                 f"outcome={info.get('outcome')} triggered={triggered}")

    if abs(rs_bonus) > 0:
        info["shaping_bonus"] = rs_bonus
    if joint and action_statuses:
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


@dataclass
class ActionDelta:
    add: List[Predicate] = field(default_factory=list)
    delete: List[Predicate] = field(default_factory=list)
    num_eff: List[Tuple[str, Dict[str, str]]] = field(default_factory=list)
    messages: List[Tuple[str, Dict[str, str]]] = field(default_factory=list)


def _apply_delta(
    env,
    delta: ActionDelta,
    info: Dict[str, Any],
    *,
    push_messages: bool = True,
) -> None:
    env.world.apply(delta.add, delta.delete)
    if env.enable_numeric:
        for expr, bind in delta.num_eff:
            apply_num_eff(env.world, expr, bind, info)
    if push_messages:
        for template, bind in delta.messages:
            rendering.push_msg(env, format_msg(env, template, bind))


def _compute_base_delta(
    env,
    act: ActionSpec,
    args: Tuple[str, ...],
    *,
    world: Optional[WorldState] = None,
) -> ActionDelta:
    base_world = world or env.world
    _, add, dele = ground_inst(env, act, args)
    dele = _expand_delete_patterns(dele, base_world.facts)
    add = _expand_add_patterns(add, base_world, act, args)
    return ActionDelta(add=add, delete=dele)


def _compute_conditional_delta(
    env,
    act: ActionSpec,
    bind: Dict[str, str],
    cond_world: WorldState,
) -> ActionDelta:
    delta = ActionDelta()
    derived_cache = {}
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
                    delta.add.append(litp)
                for d in cb.delete:
                    _, litp = ground_literal(d, b2)
                    delta.delete.append(litp)
                if env.enable_numeric:
                    for ne in cb.num_eff:
                        delta.num_eff.append((ne, b2))
                for m in cb.messages:
                    delta.messages.append((m, b2))
    return delta

def _select_outcome(env, act: ActionSpec, bind: Dict[str, str], *,
                    warn_nonpositive_total_prob: bool = False,
                    info: Optional[Dict[str, Any]] = None):
    if not getattr(act, "outcomes", None):
        return None, None, None
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
    if not valid:
        return None, None, None

    choice = None
    roll = None
    totp = None
    if env.enable_stochastic:
        totp = sum(max(0.0, float(getattr(oc, "p", 0.0))) for oc in valid)
        if warn_nonpositive_total_prob and info is not None and totp <= 0.0:
            info["stochastic_warning"] = "nonpositive_total_probability"
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

    return choice, roll, totp

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


def _validate_duration_multiplier(act: ActionSpec, bind: Dict[str, str]) -> Optional[str]:
    n_name = getattr(act, "duration_var", None)
    if not n_name:
        return None
    raw = bind.get(n_name)
    try:
        dur_multiplier = float(raw)
    except Exception:
        return f"Bad duration_var '{n_name}': {raw!r}"
    if dur_multiplier <= 0.0:
        return f"Duration multiplier '{n_name}' must be > 0."
    return None


def _resolve_action(
    env,
    name: str,
    args: Tuple[str, ...],
    *,
    display_name: Optional[str] = None,
) -> Tuple[Optional[ActionSpec], Optional[Dict[str, str]], Optional[str]]:
    display = display_name or name
    if name not in env.domain.actions:
        return None, None, f"Unknown action '{display}'"

    act: ActionSpec = env.domain.actions[name]

    expected, got = len(act.params), len(args)
    if got != expected:
        return None, None, (
            f"Arity mismatch for action '{display}': expected {expected}, got {got}"
        )

    type_err = _check_arg_types(env, act, args)
    if type_err:
        return None, None, type_err

    static_err = _check_static_effects(env, act)
    if static_err:
        return None, None, static_err

    bind = {var: val for (var, _), val in zip(act.params, args)}
    duration_err = _validate_duration_multiplier(act, bind)
    if duration_err:
        return None, None, duration_err

    return act, bind, None


def _duration_for_action(
    env,
    act: ActionSpec,
    bind: Dict[str, str],
) -> Tuple[Optional[float], Optional[str]]:
    vname = getattr(act, "duration_var", None)
    unit = getattr(act, "duration_unit", None)
    if isinstance(vname, str) and isinstance(unit, (int, float)):
        sval = bind.get(vname)
        if sval is None:
            return None, f"Bad duration var '{vname}' value: {sval!r}"
        try:
            dur = float(unit) * float(sval)
        except (TypeError, ValueError):
            return None, f"Bad duration var '{vname}' value: {sval!r}"
        return dur, None
    dur = float(act.duration) if (act.duration is not None) else env.default_duration
    return dur, None


def _collect_joint_calls(
    env,
    plan: List[Tuple[str, Tuple[str, ...]]],
    robots: List[str],
) -> Tuple[Optional[List[ActionCall]], Optional[List[float]], Optional[str]]:
    actions_by_robot: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    calls: List[ActionCall] = []
    durations: List[float] = []

    for name, args in plan:
        lname = name.lower()
        if lname == "idle":
            if len(args) != 1:
                return None, None, "Idle expects exactly one robot argument: (idle r)."
            r = args[0]
            if r not in robots:
                return None, None, f"Unknown robot '{r}' for idle."
            if r in actions_by_robot:
                return None, None, f"Multiple actions provided for robot '{r}'."
            actions_by_robot[r] = (lname, args)
            calls.append(ActionCall(act=None, name=lname, args=args))
            durations.append(0.0)
            continue

        act, bind, err = _resolve_action(env, lname, args, display_name=name)
        if err:
            return None, None, err

        r = _robot_sym_from_params(act, args)
        if not r:
            return None, None, f"Action '{name}' must include a robot parameter in joint mode."
        if r in actions_by_robot:
            return None, None, f"Multiple actions provided for robot '{r}'."
        actions_by_robot[r] = (lname, args)
        calls.append(ActionCall(act=act, name=lname, args=args, bind=bind))
        if env.enable_durations:
            dur, err = _duration_for_action(env, act, bind)
            if err:
                return None, None, err
            durations.append(dur)

    missing = [r for r in robots if r not in actions_by_robot]
    if missing:
        return None, None, f"Joint step requires one action per robot; missing for {missing}."

    return calls, durations, None

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
