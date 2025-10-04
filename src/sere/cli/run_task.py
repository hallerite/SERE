import random
import argparse
from sere.io.task_loader import load_task

DEFAULT_TASK = "tasks/kitchen/t01_one_step_steep.yaml"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--raw", action="store_true", help="Raw mode: print only system prompt once, then observations")
    args = parser.parse_args()

    # --- choose a random seed each run ---
    seed = random.randint(0, 2**32 - 1)

    env, meta = load_task(
        None,
        args.task,
        max_steps=50,
        enable_stochastic=True,     # let outcomes + NL vary
        seed=seed,                  # pass the random seed
        formatter_config=dict(
            display_nl=True,        # True => NL+PDDL everywhere; False => PDDL-only
            show_objects_in_sysprompt=True,
            show_affordances="All",
        ),
    )

    obs, info = env.reset()

    if args.raw:
        # Print only the raw system prompt if available
        sp = info.get("system_prompt")
        if sp:
            print(sp)
        # Initial observation
        print(obs)
    else:
        # Verbose/default mode
        print(f"Task: {meta['id']} — {meta['name']}  (domain: {meta.get('domain','?')})\n")

        sp = info.get("system_prompt")
        if sp:
            print("=== System Prompt ===")
            print(sp)
            print("=== End System Prompt ===\n")

        ep = info.get("episode_intro")
        if ep:
            print("=== Episode Intro ===")
            print(ep)
            print("=== End Episode Intro ===\n")

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

        if args.raw:
            # Only show the raw observation
            print(obs)
        else:
            print(f"\nReward: {r:.3f} | Outcome: {step_info.get('outcome','ongoing')}")
            for m in (step_info.get("messages") or []):
                print(f"  - {m}")
            print(obs)

        if done:
            break

if __name__ == "__main__":
    main()
