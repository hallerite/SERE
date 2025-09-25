import sys
from sere.io.task_loader import load_task

DEFAULT_TASK = "tasks/kitchen/t01_one_step_steep.yaml"

def main():
    task_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    env, meta = load_task(
        None,
        task_path,
        max_steps=50,
        enable_stochastic=False,
        formatter_config=dict(
            display_nl=True,           # True => NL+PDDL everywhere; False => PDDL-only
            show_objects_in_sysprompt=True,
            show_affordances=True,
        )
    )

    obs, info = env.reset()
    print(f"Task: {meta['id']} — {meta['name']}  (domain: {meta.get('domain','?')})\n")

    # Show the domain-only system prompt once
    sp = info.get("system_prompt")
    if sp:
        print("=== System Prompt ===")
        print(sp)
        print("=== End System Prompt ===\n")

    # Show the one-shot, instance-specific episode intro once
    ep = info.get("episode_intro")
    if ep:
        print("=== Episode Intro ===")
        print(ep)
        print("=== End Episode Intro ===\n")

    # First observation (current, volatile state)
    print("\n" + obs)

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
