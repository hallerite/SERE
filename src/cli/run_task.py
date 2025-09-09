import sys
from src.io.task_loader import load_task

DEFAULT_TASK = "tasks/kitchen/t01_one_step_steep.yaml"

def main():
    task_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    env, meta = load_task(
        None,                  # resolve domain by convention
        task_path,
        max_steps=50,
        enable_stochastic=False,
        formatter_config=dict(
            obs_mode="nl",              # "nl" | "pddl" | "both"
            show_affordances=True,        # list valid moves each turn
            include_static_in_briefing=True,
            include_problem_pddl_in_briefing=True,
            show_goal_nl=True,
            nl_max_facts=200,
        ),
    )

    obs, info = env.reset()
    print(f"Task: {meta['id']} — {meta['name']}  (domain: {meta.get('domain','?')})\n")
    # Optional: show the briefing/system prompt once
    sp = info.get("system_prompt")
    if sp:
        print("=== System Prompt / Briefing ===")
        print(sp)
        print("=== End Briefing ===\n")

    print("\n" + obs)

    # Optional: top-up energy for convenience if the fluent exists
    try:
        cur = env.world.get_fluent("energy", ("r1",))
        env.world.set_fluent("energy", ("r1",), max(cur, 10))
    except Exception:
        pass

    while True:
        try:
            move = input("\n<move> ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            return
        if not move or move.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            return

        obs, r, done, step_info = env.step(f"<move>{move}</move>")
        print(f"\nReward: {r:.3f} | Outcome: {step_info.get('outcome','ongoing')}")

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
