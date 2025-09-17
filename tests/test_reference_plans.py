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
