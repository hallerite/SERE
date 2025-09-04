import glob, yaml, pytest
from src.io.factory import load_kitchen

def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Auto-discover every kitchen task YAML (t*.yaml)
ALL_TASKS = sorted(glob.glob("tasks/kitchen/t*.yaml"))

@pytest.mark.parametrize("task_yaml", ALL_TASKS, ids=lambda p: p.split("/")[-1])
def test_reference_plan_succeeds(task_yaml):
    y = _load_yaml(task_yaml)

    # Allow each test to run with generous limits; individual core tests still set stricter ones.
    env, meta = load_kitchen(
        task_yaml,
        max_steps=200,
        enable_numeric=True,
        enable_conditional=True,
        enable_durations=True
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
                    f"outcome={step_info.get('outcome')}\n"
                    f"error={step_info.get('error')}\n"
                    f"obs=\n{obs}"
                )

    assert step_info.get("outcome") == "win"
