"""Broader agentic eval: run the PDDL sandbox against gpt-4.1-mini via Prime API.

7 domains x 3 problems x 2 rollouts = 42 episodes.
Computes pass@1 per domain and overall.
"""

import json
import math
import os
import sys
import time
import traceback
from importlib.resources import files as pkg_files

from openai import OpenAI

from sere.io.pddl_loader import load_agentic_task

API_KEY = os.environ.get("PRIME_API_KEY", "").strip()
TEAM_ID = os.environ.get("PRIME_TEAM_ID", "").strip()
BASE_URL = "https://api.pinference.ai/api/v1"
MODEL = "openai/gpt-4.1-mini"
MAX_TURNS = 20

# -- Config --
DOMAINS = [
    "blocksworld",
    "gripper",
    "ferry",
    "miconic",
    "logistics",
    "transport",
    "sokoban",
]
PROBLEMS_PER_DOMAIN = 3
ROLLOUTS_PER_PROBLEM = 2


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k from n total rollouts with c correct.

    Uses the unbiased estimator: 1 - comb(n-c, k) / comb(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def run_one(domain_name: str, problem_idx: int = 0):
    """Run a single agentic episode. Returns (solved: bool, info: dict)."""
    pddl_root = pkg_files("sere.assets.pddl")
    domain_dir = pddl_root / domain_name
    probs = sorted((domain_dir / "problems").iterdir())
    if problem_idx >= len(probs):
        print(f"Only {len(probs)} problems in {domain_name}")
        return False, {"error": "problem_idx out of range"}

    env, meta = load_agentic_task(str(domain_dir), str(probs[problem_idx]))
    problem_name = meta["name"]
    print(f"Problem: {problem_name} ({domain_name})")
    print(f"Goal: {env.goal_expr}")
    print()

    headers = {}
    if TEAM_ID:
        headers["X-Prime-Team-ID"] = TEAM_ID
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, default_headers=headers)

    messages = [
        {"role": "system", "content": env.system_prompt()},
        {"role": "user", "content": "Solve this PDDL planning problem. Read the domain and problem files, write a plan, and validate it."},
    ]
    tools = env.tool_schemas()

    t0 = time.time()
    total_turns = 0

    for turn in range(MAX_TURNS):
        total_turns = turn + 1
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            print(f"[turn {turn+1}] API error: {e}")
            break

        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if not msg.tool_calls:
            print(f"[turn {turn+1}] assistant: {msg.content[:200] if msg.content else '(empty)'}")
            if env.solved:
                break
            messages.append({"role": "user", "content": "Use the tools to write and validate your plan."})
            continue

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            feedback, done = env.handle_tool_call(fn_name, fn_args)

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

    elapsed = time.time() - t0
    solved = env.solved
    attempts = env.attempts

    print()
    print(f"Result: {'SOLVED' if solved else 'FAILED'}")
    print(f"Attempts: {attempts}/{env.max_attempts}  Turns: {total_turns}  Time: {elapsed:.1f}s")

    info = {
        "problem": problem_name,
        "domain": domain_name,
        "problem_idx": problem_idx,
        "solved": solved,
        "attempts": attempts,
        "turns": total_turns,
        "elapsed_s": round(elapsed, 2),
    }

    env.cleanup()
    return solved, info


def main():
    if not API_KEY:
        print("Set PRIME_API_KEY environment variable")
        sys.exit(1)

    print(f"Model: {MODEL}")
    print(f"Domains: {DOMAINS}")
    print(f"Problems/domain: {PROBLEMS_PER_DOMAIN}, Rollouts/problem: {ROLLOUTS_PER_PROBLEM}")
    print(f"Total episodes: {len(DOMAINS) * PROBLEMS_PER_DOMAIN * ROLLOUTS_PER_PROBLEM}")
    print()

    # Detailed per-episode results
    all_episodes = []
    # For pass@1 computation: domain -> problem_idx -> list of bools
    rollout_results = {}

    episode_num = 0
    total_episodes = len(DOMAINS) * PROBLEMS_PER_DOMAIN * ROLLOUTS_PER_PROBLEM

    for domain in DOMAINS:
        rollout_results[domain] = {}
        for pidx in range(PROBLEMS_PER_DOMAIN):
            rollout_results[domain][pidx] = []
            for rollout in range(ROLLOUTS_PER_PROBLEM):
                episode_num += 1
                key = f"{domain}/p{pidx}/r{rollout}"
                print(f"\n{'='*60}")
                print(f"Episode {episode_num}/{total_episodes}: {key}")
                print(f"{'='*60}")
                try:
                    solved, info = run_one(domain, pidx)
                    info["rollout"] = rollout
                    info["key"] = key
                    all_episodes.append(info)
                    rollout_results[domain][pidx].append(solved)
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()
                    all_episodes.append({
                        "key": key,
                        "domain": domain,
                        "problem_idx": pidx,
                        "rollout": rollout,
                        "solved": False,
                        "error": str(e),
                    })
                    rollout_results[domain][pidx].append(False)

    # -- Compute pass@1 per domain and overall --
    domain_pass1 = {}
    all_problem_pass1 = []

    for domain in DOMAINS:
        problem_pass1_values = []
        for pidx in range(PROBLEMS_PER_DOMAIN):
            results_list = rollout_results[domain][pidx]
            n = len(results_list)
            c = sum(results_list)
            p1 = pass_at_k(n, c, 1)
            problem_pass1_values.append(p1)
            all_problem_pass1.append(p1)

        domain_pass1[domain] = sum(problem_pass1_values) / len(problem_pass1_values) if problem_pass1_values else 0.0

    overall_pass1 = sum(all_problem_pass1) / len(all_problem_pass1) if all_problem_pass1 else 0.0

    # Build summary
    summary = {
        "model": MODEL,
        "domains": DOMAINS,
        "problems_per_domain": PROBLEMS_PER_DOMAIN,
        "rollouts_per_problem": ROLLOUTS_PER_PROBLEM,
        "total_episodes": total_episodes,
        "pass_at_1_per_domain": {d: round(v, 4) for d, v in domain_pass1.items()},
        "overall_pass_at_1": round(overall_pass1, 4),
        "raw_solve_rate": round(sum(1 for ep in all_episodes if ep.get("solved")) / len(all_episodes), 4) if all_episodes else 0.0,
    }

    full_output = {
        "summary": summary,
        "episodes": all_episodes,
    }

    # Save to file
    output_path = "/home/ubuntu/SERE/outputs/gpt41mini_agentic_eval.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Total episodes: {total_episodes}")
    print()
    print("pass@1 per domain:")
    for domain in DOMAINS:
        p1 = domain_pass1[domain]
        # Also show raw solve counts
        domain_episodes = [ep for ep in all_episodes if ep.get("domain") == domain]
        solved_count = sum(1 for ep in domain_episodes if ep.get("solved"))
        total_count = len(domain_episodes)
        print(f"  {domain:15s}  pass@1={p1:.4f}  ({solved_count}/{total_count} raw solves)")
    print()
    print(f"Overall pass@1: {overall_pass1:.4f}")
    print(f"Raw solve rate:  {summary['raw_solve_rate']:.4f}")


if __name__ == "__main__":
    main()
