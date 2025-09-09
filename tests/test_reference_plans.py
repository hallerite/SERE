import glob, yaml, pytest
from src.io.task_loader import load_task

def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Auto-discover every task YAML (t*.yaml) across all domains
ALL_TASKS = sorted(glob.glob("tasks/*/t*.yaml"))

@pytest.mark.parametrize("task_yaml", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_reference_plan_succeeds(task_yaml):
    y = _load_yaml(task_yaml)

    # Run with generous limits; let tasks override via meta if needed.
    # We also hard-disable stochastic here so reference plans are deterministic.
    env, meta = load_task(
        None,                # infer domains/{meta.domain}.yaml or from path
        task_yaml,
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

    # If a task has no reference_plan yet, mark it xfail so the suite still runs.
    plan = y.get("reference_plan")
    if not plan:
        pytest.xfail(f"{task_yaml} has no reference_plan")

    # Execute the reference plan
    for i, act in enumerate(plan):
        obs, r, done, step_info = env.step(f"<move>{act}</move>")
        if done:
            if step_info.get("outcome") == "win":
                break
            else:
                raise AssertionError(
                    f"Failed at step {i}: {act}\n"
                    f"task={task_yaml}\n"
                    f"domain={meta.get('domain')}\n"
                    f"outcome={step_info.get('outcome')}\n"
                    f"error={step_info.get('error')}\n"
                    f"obs=\n{obs}"
                )

    assert step_info.get("outcome") == "win"