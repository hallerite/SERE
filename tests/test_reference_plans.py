import pytest
from importlib.resources import files as pkg_files

from sere.io.task_loader import load_task
from sere.core.pddl_env.planning import execute_plan, execute_joint, parse_actions


def _iter_task_ids():
    base = pkg_files("sere.assets.tasks")
    out = []
    for domain_dir in base.iterdir():
        if not domain_dir.is_dir():
            continue
        for entry in domain_dir.iterdir():
            if entry.is_file() and entry.name.startswith("t") and entry.name.endswith(".yaml"):
                out.append(f"{domain_dir.name}/{entry.name}")
    return sorted(out)


ALL_TASKS = _iter_task_ids()

FAST_FORMATTER = {
    "show_affordances": False,
    "show_footer": False,
    "show_messages": False,
    "display_nl": False,
}


def _load_task_fast(task_id: str, **kwargs):
    formatter_config = dict(FAST_FORMATTER)
    formatter_config.update(kwargs.pop("formatter_config", {}) or {})
    return load_task(None, task_id, formatter_config=formatter_config, **kwargs)


@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_has_single_success_goal_rule_under_termination(task_id: str):
    env, meta = _load_task_fast(task_id)

    term_meta = meta.get("termination")
    assert isinstance(term_meta, list), f"{task_id}: `termination:` key missing or not a list"

    rules = getattr(env, "termination_rules", [])
    assert isinstance(rules, list) and rules, f"{task_id}: no termination rules parsed into env"

    success = [r for r in rules if str(r.get("outcome", "")).lower() == "success"]
    assert len(success) == 1, f"{task_id}: expected exactly ONE success rule, found {len(success)}"
    assert str(success[0].get("when", "")).strip(), f"{task_id}: success rule must specify `when`"


@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_reference_plan_succeeds_and_reward_matches(task_id: str):
    env, meta = _load_task_fast(task_id, enable_stochastic=False, seed=0)
    env.reset()

    plan = meta.get("reference_plan") or []
    if not plan:
        pytest.xfail(f"{task_id} has no reference_plan")

    plan_str = "".join(plan)  # "(op ...)(op ...)"
    plan_seq = parse_actions(plan_str)
    if env.multi_agent:
        obs, total_reward, done, info = execute_joint(env, plan_seq)
    else:
        obs, total_reward, done, info = execute_plan(env, plan_seq, atomic=True)

    assert done, f"{task_id}: plan did not terminate"
    assert info.get("outcome") == "success", f"{task_id}: plan outcome = {info.get('outcome')}"

    # Baseline: step penalties + termination reward
    n_steps = int(info.get("steps_executed", len(plan_seq)))
    term_rules = [r for r in env.termination_rules if str(r.get("outcome", "")).lower() == "success"]
    assert len(term_rules) == 1
    success_reward = float(term_rules[0].get("reward", 0.0))
    expected_baseline = env.step_penalty * n_steps + success_reward

    # Non-atomic recompute of shaping on a fresh env with the same plan
    env2, _ = _load_task_fast(task_id, enable_stochastic=False, seed=0)
    env2.reset()
    if env2.multi_agent:
        _, _, _, info2 = execute_joint(env2, plan_seq)
    else:
        _, _, _, info2 = execute_plan(env2, plan_seq, atomic=False)
    observed_shaping = float(info2.get("shaping_bonus_total", 0.0))

    expected_total = expected_baseline + observed_shaping
    assert total_reward == pytest.approx(expected_total, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------
# Additional invariant and energy sanity tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_reset_has_no_invariant_violations(task_id: str):
    """Reset should not violate world invariants or produce empty obs."""
    env, _ = _load_task_fast(task_id)
    obs, _ = env.reset()
    assert isinstance(obs, str) and obs.strip(), "Empty observation after reset"
    errs = env.world.validate_invariants()
    assert not errs, f"Invariants violated at reset: {errs}"


@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_initial_energy_not_exceeding_cap(task_id: str):
    """If a battery-cap exists, initial energy must be within [0, cap]."""
    env, _ = _load_task_fast(task_id)
    env.reset()
    robots = [sym for sym, types in env.world.objects.items() if "robot" in (types or set())]
    if not robots:
        pytest.skip("No robots")
    violations = []
    for r in robots:
        cap_is_set = ("battery-cap", (r,)) in env.world.fluents
        if not cap_is_set:
            continue
        cap = env.world.get_fluent("battery-cap", (r,))
        e = env.world.get_fluent("energy", (r,))
        if e < -1e-9 or e - cap > 1e-9:
            violations.append((r, e, cap))
    if not violations and not any(("battery-cap", (r,)) in env.world.fluents for r in robots):
        pytest.skip("No battery-cap defined")
    assert not violations, f"Initial energy exceeds cap or is negative: {violations}"


@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_energy_is_clamped_after_first_step_if_above_cap(task_id: str):
    """Deliberately set energy > cap, take a step, ensure clamp occurred."""
    env, _ = _load_task_fast(task_id)
    env.reset()
    robots = [sym for sym, types in env.world.objects.items() if "robot" in (types or set())]
    if not robots:
        pytest.skip("No robots")
    r = robots[0]
    cap_is_set = ("battery-cap", (r,)) in env.world.fluents
    if not cap_is_set:
        pytest.skip("No battery-cap")
    cap = env.world.get_fluent("battery-cap", (r,))
    env.world.set_fluent("energy", (r,), cap + 10.0)
    if env.multi_agent:
        joint = "".join(f"(idle {sym})" for sym in robots)
        env.step(joint)
    else:
        env.step("(wait 1)")
    assert env.world.get_fluent("energy", (r,)) <= cap + 1e-9


@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_recharge_does_not_exceed_cap_when_possible(task_id: str):
    """One recharge should not overflow battery-cap (clamp applies)."""
    env, _ = _load_task_fast(task_id)
    env.reset()
    robots = [sym for sym, types in env.world.objects.items() if "robot" in (types or set())]
    if not robots:
        pytest.skip("No robots")
    r = robots[0]
    cap_is_set = ("battery-cap", (r,)) in env.world.fluents
    if not cap_is_set:
        pytest.skip("No battery-cap")
    cap = env.world.get_fluent("battery-cap", (r,))
    locs = [a[1] for (pred, a) in env.world.facts if pred == "at" and a[0] == r]
    if len(locs) != 1:
        pytest.skip("Ambiguous robot location")
    l = locs[0]
    if ("has-charger", (l,)) not in env.static_facts:
        pytest.skip("No charger at robot's location")
    start_e = max(0.0, cap - 0.5)
    env.world.set_fluent("energy", (r,), start_e)
    if env.multi_agent:
        joint = "".join(
            f"(recharge {r} {l})" if sym == r else f"(idle {sym})"
            for sym in robots
        )
        env.step(joint)
    else:
        env.step(f"(recharge {r} {l})")
    e_after = env.world.get_fluent("energy", (r,))
    assert e_after <= cap + 1e-9, f"Recharge overflowed cap: after={e_after}, cap={cap}"
    assert e_after >= start_e, "Recharge should not reduce energy"
