import sys
from ..io.factory import load_kitchen

def main():
    task_path = sys.argv[1] if len(sys.argv) > 1 else "tasks/kitchen/t01_one_step_steep.yaml"
    env, meta = load_kitchen(task_path, max_steps=50)
    obs, info = env.reset()
    print(f"Task: {meta['id']} — {meta['name']}")
    print(info["problem_pddl"])
    print("\n" + obs)

    while True:
        try:
            move = input("\n<move> ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            return
        obs, r, done, info = env.step(f"<move>{move}</move>")
        print(f"\nReward: {r:.3f} | Outcome: {info.get('outcome','ongoing')}")
        print(obs)
        if done: break

if __name__ == "__main__":
    main()
