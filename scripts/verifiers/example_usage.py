#!/usr/bin/env python3
"""
Example: Using SERE with Verifiers for RL training.

This script demonstrates how to:
1. Load SERE tasks as a Verifiers environment
2. Inspect the dataset
3. Run basic evaluation (mock)
"""

from sere.integrations.verifiers import (
    load_environment,
    discover_tasks,
    get_available_domains,
)


def main():
    print("=" * 60)
    print("SERE Verifiers Integration - Example Usage")
    print("=" * 60)

    # 1. Discover available tasks
    print("\n1. Discovering SERE tasks...")
    domains = get_available_domains()
    print(f"   Available domains: {sorted(domains)}")

    kitchen_tasks = discover_tasks(domains=["kitchen"], num_tasks_per_domain=3)
    print(f"   First 3 kitchen tasks: {kitchen_tasks}")

    # 2. Load environment
    print("\n2. Loading environment...")
    env = load_environment(
        domains=["kitchen"],
        num_tasks_per_domain=3,
        episodes_per_task=2,  # 2 training episodes per task
        eval_episodes_per_task=1,  # 1 eval episode per task
        max_episode_steps=20,  # Limit episode length
    )

    print(f"   ✓ Environment loaded")
    print(f"   - Training episodes: {len(env.dataset)}")
    print(f"   - Eval episodes: {len(env.eval_dataset)}")
    print(f"   - Max steps per episode: {env.max_turns}")

    # 3. Inspect dataset
    print("\n3. Inspecting dataset...")
    sample = env.dataset[0]
    print(f"   First training example:")
    print(f"   - Question (first 200 chars): {sample['question'][:200]}...")
    print(f"   - Answer: {sample['answer']}")

    # 4. Check system prompt
    print("\n4. System prompt:")
    print(f"   {env.system_prompt[:200]}...")

    # 5. Example: How to evaluate (commented out - requires model)
    print("\n5. Evaluation example (commented out):")
    print("""
    # To run actual evaluation, you need an OpenAI-compatible server:

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="http://localhost:8000/v1")

    results = env.evaluate(
        client=client,
        model="gpt-4o-mini",
        num_rollouts=10,
    )

    print(f"Average reward: {sum(results['reward']) / len(results['reward'])}")
    print(f"Success rate: {sum(r > 0 for r in results['reward']) / len(results['reward'])}")
    """)

    # 6. Multi-agent example
    print("\n6. Multi-agent task example:")
    multi_agent_tasks = discover_tasks(domains=["kitchen"])
    multi_agent_tasks = [t for t in multi_agent_tasks if "multi_agent" in t][:1]

    if multi_agent_tasks:
        env_multi = load_environment(
            task_paths=multi_agent_tasks,
            episodes_per_task=1,
        )
        sample = env_multi.dataset[0]
        print(f"   Multi-agent task loaded: {multi_agent_tasks[0]}")
        print(f"   - Question (first 200 chars): {sample['question'][:200]}...")
    else:
        print("   (No multi-agent tasks found)")

    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
