"""Smoke test: run the RML agentic env with an LLM via Prime API."""
import json
import os
import sys

from openai import OpenAI

from sere.rml.problems import load_problem
from sere.rml.rml_env import AgenticRMLEnv

API_KEY = os.environ.get("PRIME_API_KEY", "").strip()
if not API_KEY:
    kf = os.path.expanduser("~/prime_api_key_inference.txt")
    if os.path.exists(kf):
        API_KEY = open(kf).read().strip()
BASE_URL = "https://api.pinference.ai/api/v1"
TEAM_ID = os.environ.get("PRIME_TEAM_ID", "").strip()
if not TEAM_ID:
    cfg = os.path.expanduser("~/.prime/config.json")
    if os.path.exists(cfg):
        TEAM_ID = json.load(open(cfg)).get("team_id", "")

MODEL = "openai/gpt-4.1-mini"
PROBLEM = sys.argv[1] if len(sys.argv) > 1 else "wall_gap"
MAX_TURNS = 30

headers = {}
if TEAM_ID:
    headers["X-Prime-Team-ID"] = TEAM_ID
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, default_headers=headers)

# Setup
problem = load_problem(PROBLEM)
env = AgenticRMLEnv(problem=problem, max_attempts=3)
print(f"=== RML Eval: {PROBLEM} with {MODEL} ===")
print(f"Goal: {problem.start} → {problem.goal}")

messages = [
    {"role": "system", "content": env.system_prompt()},
    {"role": "user", "content": "Read the problem and plan a route. Submit when ready."},
]

for turn in range(MAX_TURNS):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=env.tool_schemas,
        max_tokens=2048,
    )
    choice = resp.choices[0]
    msg = choice.message

    # Build assistant message
    asst = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        asst["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    messages.append(asst)

    if msg.content:
        print(f"\n[Turn {turn+1}] Assistant: {msg.content[:200]}")

    if not msg.tool_calls:
        if choice.finish_reason == "stop":
            print(f"[Turn {turn+1}] Model stopped without tool call.")
            break
        continue

    # Process all tool calls
    done = False
    for tc in msg.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}

        feedback, is_done = env.handle_tool_call(name, args)
        print(f"[Turn {turn+1}] {name}({json.dumps(args)[:80]}) → {feedback[:120]}")

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": feedback,
        })
        if is_done:
            done = True

    if done:
        break

print(f"\n=== Result: solved={env.solved}, attempts={env.attempts} ===")
env.cleanup()
