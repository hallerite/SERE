import pytest
from importlib.resources import files as pkg_files

from sere.io.task_loader import load_task
from sere.core.semantics import eval_clause


def _iter_task_ids():
    base = pkg_files("sere.assets.tasks")
    out = []
    for domain_dir in base.iterdir():
        if not domain_dir.is_dir():
            continue
        for entry in domain_dir.iterdir():
            name = entry.name
            if entry.is_file() and name.startswith("t") and name.endswith(".yaml"):
                out.append(f"{domain_dir.name}/{name}")
    return sorted(out)


ALL_TASKS = _iter_task_ids()


@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_reference_plan_succeeds_and_reward_matches(task_id: str):
    env, meta = load_task(None, task_id)

    obs, info = env.reset()
    try:
        env.world.set_fluent("energy", ("r1",), 10)
    except Exception:
        pass

    plan = meta.get("reference_plan") or []
    if not plan:
        pytest.xfail(f"{task_id} has no reference_plan")

    total_reward = 0.0
    shaping_total_observed = 0.0
    n_steps_executed = 0

    # dynamic expected shaping mirror for instant mode
    rs_expected_dynamic = 0.0
    rs_seen = set()  # indices for once=True
    rs_milestones = list(getattr(env, "rs_milestones", []) or [])
    rs_mode = getattr(env, "rs_mode", "instant")

    step_info = {}
    for i, act in enumerate(plan):
        obs, r, done, step_info = env.step(f"<move>{act}</move>")
        total_reward += r
        shaping_total_observed += float(step_info.get("shaping_bonus", 0.0))
        n_steps_executed += 1

        # compute expected shaping payout online to handle non-monotone exprs
        if rs_milestones and rs_mode == "instant":
            for j, (expr, rew, once) in enumerate(rs_milestones):
                try:
                    hit = eval_clause(env.world, env.static_facts, expr, bind={}, enable_numeric=True)
                except Exception:
                    hit = False
                if once:
                    if hit and j not in rs_seen:
                        rs_expected_dynamic += float(rew)
                        rs_seen.add(j)
                else:
                    if hit:
                        rs_expected_dynamic += float(rew)

        if done:
            if step_info.get("outcome") == "success":
                break
            else:
                raise AssertionError(
                    f"Failed at step {i}: {act}\n"
                    f"task={task_id}\n"
                    f"domain={meta.get('domain')}\n"
                    f"outcome={step_info.get('outcome')}\n"
                    f"error={step_info.get('error')}\n"
                    f"obs=\n{obs}"
                )

    assert step_info.get("outcome") == "success"

    # Baseline reward check
    baseline = env.step_penalty * n_steps_executed + env.goal_reward
    assert total_reward == pytest.approx(baseline + shaping_total_observed, rel=1e-12, abs=1e-12)

    # Instant-mode shaping should match our dynamic mirror (not final-state truth!)
    if rs_milestones and rs_mode == "instant":
        assert shaping_total_observed == pytest.approx(rs_expected_dynamic, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------
# Additional invariant and energy sanity tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_reset_has_no_invariant_violations(task_id: str):
    """Reset should not violate world invariants or produce empty obs."""
    env, _ = load_task(None, task_id)
    obs, _ = env.reset()
    assert isinstance(obs, str) and obs.strip(), "Empty observation after reset"
    errs = env.world.validate_invariants()
    assert not errs, f"Invariants violated at reset: {errs}"


@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_initial_energy_not_exceeding_cap(task_id: str):
    """If a battery-cap exists, initial energy must be within [0, cap]."""
    env, _ = load_task(None, task_id)
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
    env, _ = load_task(None, task_id)
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
    obs, rwd, done, info = env.step("<move>(wait 1)</move>")
    assert env.world.get_fluent("energy", (r,)) <= cap + 1e-9


@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_recharge_does_not_exceed_cap_when_possible(task_id: str):
    """One recharge should not overflow battery-cap (clamp applies)."""
    env, _ = load_task(None, task_id)
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
    env.step(f"<move>(recharge {r} {l})</move>")
    e_after = env.world.get_fluent("energy", (r,))
    assert e_after <= cap + 1e-9, f"Recharge overflowed cap: after={e_after}, cap={cap}"
    assert e_after >= start_e, "Recharge should not reduce energy"
