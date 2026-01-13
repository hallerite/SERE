#!/usr/bin/env python3
"""Quick test script for long-horizon tasks"""

from sere.io.task_loader import load_task
from sere.core.pddl_env.run_mode import RunMode

def test_task(task_path: str):
    """Test a task by running its reference plan"""
    print(f"\n{'='*60}")
    print(f"Testing: {task_path}")
    print('='*60)

    try:
        # Load task
        env, meta = load_task(
            domain_path=None,
            task_path=task_path,
            run_mode=RunMode.OPEN_LOOP,
            enable_stochastic=False,
        )

        # Reset
        obs, info = env.reset(seed=0)
        print(f"✓ Task loaded successfully")
        print(f"  Multi-agent: {env.multi_agent}")
        print(f"  Max steps: {env.max_steps}")

        # Get reference plan from meta
        ref_plan = meta.get("reference_plan")
        if not ref_plan:
            print("✗ No reference plan found")
            return False

        print(f"  Reference plan steps: {len(ref_plan)}")

        # Execute plan
        plan_str = "".join(ref_plan)
        obs, reward, done, info = env.step(plan_str)

        # Check outcome
        outcome = info.get("outcome", "unknown")
        steps_executed = info.get("steps_executed", 0)

        print(f"\n  Steps executed: {steps_executed}")
        print(f"  Outcome: {outcome}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")

        if outcome == "success":
            print(f"\n✓ PASS: Task completed successfully!")
            return True
        else:
            print(f"\n✗ FAIL: Task did not complete successfully")
            if "error" in info:
                print(f"  Error: {info['error']}")
            return False

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    tasks = [
        "kitchen/t15_restaurant_shift.yaml",
        "assembly/t07_build_complex_device.yaml",
    ]

    results = {}
    for task in tasks:
        results[task] = test_task(task)

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for task, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {task}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All {len(results)} tasks passed!")
        exit(0)
    else:
        failed = sum(1 for p in results.values() if not p)
        print(f"\n✗ {failed}/{len(results)} tasks failed")
        exit(1)
