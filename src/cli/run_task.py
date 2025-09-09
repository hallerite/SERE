import sys
from src.io.task_loader import load_task

DEFAULT_TASK = "tasks/kitchen/t01_one_step_steep.yaml"

def main():
    task_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    env, meta = load_task(
        None,           # resolve domains/{meta.domain}.yaml by convention
        task_path,
        max_steps=50,
        enable_stochastic=False,
    )

    obs, info = env.reset()
    print(f"Task: {meta['id']} — {meta['name']}  (domain: {meta.get('domain','?')})")
    print(info["problem_pddl"])
    print("\n" + obs)

    # Optional: top-up energy for convenience if the fluent exists
    try:
        env.world.set_fluent("energy", ("r1",), max(env.world.get_fluent("energy", ("r1",)), 10))
    except Exception:
        pass

    while True:
        try:
            move = input("\n<move> ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            return
        if not move:
            continue
        obs, r, done, step_info = env.step(f"<move>{move}</move>")
        print(f"\nReward: {r:.3f} | Outcome: {step_info.get('outcome','ongoing')}")
        # Surface any agent messages (QC etc.)
        msgs = step_info.get("messages") or []
        if msgs:
            print("Messages:")
            for m in msgs:
                print(f"  - {m}")
        print(obs)
        if done:
            break

if __name__ == "__main__":
    main()
