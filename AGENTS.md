# Agentic Planning in SERE

This document describes how LLM agents solve PDDL planning problems in SERE.

---

## Overview

SERE provides a **miniSWE-style sandbox** where an LLM agent uses familiar coding tools to solve planning problems. The agent works in a temp directory containing PDDL files and iterates on a plan using tool calls until it passes validation.

This is the primary mode for evaluation and RL training.

---

## How It Works

### 1. Workspace

Each episode creates a sandboxed workspace:

```
workspace/
  domain.pddl   -- planning domain definition (read-only)
  problem.pddl  -- problem instance with init state + goal (read-only)
  plan.pddl     -- the agent writes its solution here
```

### 2. Tools

The agent has 6 tools, matching a standard coding agent:

| Tool | Purpose |
|------|---------|
| `bash(command)` | Run shell commands (e.g., `grep`, `wc -l`, `diff`) |
| `read_file(path)` | Read any file in the workspace |
| `write_file(path, content)` | Create or overwrite a file |
| `str_replace(path, old_str, new_str)` | Edit a file (old_str must be unique) |
| `validate(up_to_step?)` | Validate plan.pddl -- full validation = submission |
| `simulate(up_to_step?)` | Execute plan.pddl and show resulting world state |

### 3. Typical Agent Loop

```
1. read_file("domain.pddl")     -- understand the actions and types
2. read_file("problem.pddl")    -- understand the objects, init state, and goal
3. write_file("plan.pddl", ...) -- write initial plan attempt
4. validate(up_to_step=5)       -- check first 5 steps (free, no attempt used)
5. str_replace("plan.pddl", ..) -- fix errors
6. validate()                   -- full submission (counts as attempt)
```

### 4. Validation Rules

- **Full validate** (no `up_to_step`, or `up_to_step >= total steps`): counts as a **submission attempt** and can set `solved=True`.
- **Partial validate** (`up_to_step < total steps`): free, returns diagnostics but doesn't count.
- **Simulate**: always free, shows world state after N steps.
- **Max attempts**: default 8 full submissions per episode.

### 5. Guards

- `domain.pddl` and `problem.pddl` are **read-only**. The `write_file` and `str_replace` tools reject writes to these files. If `bash` modifies them, they're restored immediately.
- The init state is **copied fresh** for each validation call -- the agent cannot accidentally corrupt it.
- The goal expression is **immutable**.

---

## Plan Format

Plans are written as grounded PDDL actions, one per line:

```
(pick-up A)
(stack A B)
(pick-up C)
(stack C D)
```

Action names and arguments must match the domain definition exactly.

---

## Using the Agentic Environment

### Direct (Python)

```python
from sere.io.pddl_loader import load_agentic_task

env, meta = load_agentic_task("path/to/blocksworld", "path/to/instance-1.pddl")

# For LLM integration
system_prompt = env.system_prompt()
tools = env.tool_schemas()

# Dispatch each tool call from the LLM
feedback, done = env.handle_tool_call("read_file", {"path": "domain.pddl"})
feedback, done = env.handle_tool_call("write_file", {
    "path": "plan.pddl",
    "content": "(pick-up A)\n(stack A B)\n",
})
feedback, done = env.handle_tool_call("validate", {})

# Check result
print(env.solved)     # True if goal reached
print(env.attempts)   # number of full submissions used
env.cleanup()         # remove temp workspace
```

### Verifiers Framework

```python
from integrations.verifiers import load_agentic_environment

env = load_agentic_environment(
    domains=["blocksworld", "gripper", "ferry"],
    num_tasks_per_domain=5,
    max_attempts=8,
)
```

### Prime CLI

```bash
prime eval run integrations.verifiers.vf_sere:load_agentic_environment \
  -m openai/gpt-4.1-mini -n 10 -r 1 --debug
```

---

## Validation Feedback

When validation fails, the agent gets detailed diagnostics:

```
Plan failed at step 3 of 8.

Action: (stack C A)
Error: Preconditions not satisfied for (stack C A)
Unsatisfied preconditions:
  - (clear A): false

State after step 2:
  (clear B)
  (clear C)
  (on B A)
  (ontable A)
  (ontable C)
  (handempty)
Goal satisfied: False
```

This tells the agent exactly which precondition failed and shows the world state, so it can fix the plan.

---

## Gym-Style API (Alternative)

For step-by-step RL interaction without the agentic sandbox:

```python
from sere.io.task_loader import load_task

env, meta = load_task(
    "src/sere/assets/pddl/blocksworld",
    "src/sere/assets/pddl/blocksworld/problems/instance-1.pddl",
)

obs, info = env.reset()
obs, reward, done, info = env.step("(pick-up A)")
```

Run modes: `INTERACTIVE` (one action per step), `BATCH` (multiple actions), `OPEN_LOOP` (full plan, no intermediate feedback).

---

## Multi-Agent Tasks

YAML domains (kitchen, assembly) support multi-agent tasks where multiple robots act simultaneously. Each step requires one action per robot:

```
(move r1 kitchen pantry)(pick-up r2 mug1)
```

Use `(idle r)` for no-op. Multi-agent tasks are available through the Gym-style API and the pure PDDL verifiers environment.
