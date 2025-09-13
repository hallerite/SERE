import pytest
from importlib.resources import files as pkg_files

from sere.io.task_loader import load_task


def _iter_task_ids():
    """
    Enumerate packaged tasks under sere.assets.tasks as logical IDs:
      e.g. 'kitchen/t01_one_step_steep.yaml'
    """
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


# Auto-discover every task YAML (t*.yaml) across all domains (packaged)
ALL_TASKS = _iter_task_ids()


@pytest.mark.parametrize("task_id", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_reference_plan_succeeds(task_id: str):
    # Run with generous limits; let tasks override via meta if needed.
    # We also hard-disable stochastic here so reference plans are deterministic.
    env, meta = load_task(
        None,                # infer domain yaml or from path
        task_id,
        max_steps=200,
        enable_numeric=True,
        enable_conditional=True,
        enable_durations=True,
        enable_stochastic=False,
    )

    # Reset and set sane default energy unless the task explicitly wants to test low energy.
    obs, info = env.reset()
    try:
        env.world.set_fluent("energy", ("r1",), 10)
    except Exception:
        pass  # ok if energy fluent not present

    # Reference plan comes from loader meta; don't open YAML directly
    plan = meta.get("reference_plan") or []
    if not plan:
        pytest.xfail(f"{task_id} has no reference_plan")

    # Execute the reference plan
    step_info = {}
    for i, act in enumerate(plan):
        obs, r, done, step_info = env.step(f"<move>{act}</move>")
        if done:
            if step_info.get("outcome") == "win":
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

    assert step_info.get("outcome") == "win"
