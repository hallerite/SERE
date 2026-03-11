"""Quick smoke test: run the agentic sandbox against a real LLM via API."""

import json
import os
import sys
from importlib.resources import files as pkg_files

from openai import OpenAI

from sere.io.pddl_loader import load_agentic_task

API_KEY = os.environ.get("PRIME_API_KEY", "").strip()
BASE_URL = "https://api.pinference.ai/api/v1"
MODEL = "openai/gpt-4.1"
MAX_TURNS = 20


def run_one(domain_name: str, problem_idx: int = 0):
    """Run a single agentic episode."""
    pddl_root = pkg_files("sere.assets.pddl")
    domain_dir = pddl_root / domain_name
    probs = sorted((domain_dir / "problems").iterdir())
    if problem_idx >= len(probs):
        print(f"Only {len(probs)} problems in {domain_name}")
        return

    env, meta = load_agentic_task(str(domain_dir), str(probs[problem_idx]))
    print(f"Problem: {meta['name']} ({domain_name})")
    print(f"Goal: {env.goal_expr}")
    print()

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [
        {"role": "system", "content": env.system_prompt()},
        {"role": "user", "content": "Solve this PDDL planning problem. Read the domain and problem files, write a plan, and validate it."},
    ]
    tools = env.tool_schemas()

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        # Add assistant message to history
        messages.append(msg.model_dump())

        if not msg.tool_calls:
            # Model sent text without tool calls
            print(f"[turn {turn+1}] assistant: {msg.content[:200] if msg.content else '(empty)'}")
            if env.solved:
                break
            # Nudge
            messages.append({"role": "user", "content": "Use the tools to write and validate your plan."})
            continue

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            feedback, done = env.handle_tool_call(fn_name, fn_args)

            # Truncate long outputs for display
            display = feedback[:200] + "..." if len(feedback) > 200 else feedback
            print(f"[turn {turn+1}] {fn_name}({json.dumps(fn_args)[:80]}) -> {display}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": feedback,
            })

            if done:
                break

        if env.solved or (env.attempts >= env.max_attempts):
            break

    print()
    print(f"Result: {'SOLVED' if env.solved else 'FAILED'}")
    print(f"Attempts: {env.attempts}/{env.max_attempts}")
    env.cleanup()
    return env.solved


def main():
    if not API_KEY:
        print("Set PRIME_API_KEY environment variable")
        sys.exit(1)

    domains = ["blocksworld", "gripper", "ferry"]
    results = {}

    for domain in domains:
        for i in range(2):  # 2 problems per domain
            key = f"{domain}/p{i}"
            print(f"\n{'='*60}")
            print(f"Running: {key}")
            print(f"{'='*60}")
            try:
                solved = run_one(domain, i)
                results[key] = solved
            except Exception as e:
                print(f"Error: {e}")
                results[key] = False

    print(f"\n{'='*60}")
    print("Summary:")
    for key, solved in results.items():
        print(f"  {key}: {'PASS' if solved else 'FAIL'}")
    total = sum(1 for v in results.values() if v)
    print(f"\n  {total}/{len(results)} solved")


if __name__ == "__main__":
    main()
