# Symbolic Embodied Reasoning Environments (SERE)

**SERE** is a lightweight framework for building **symbolic planning environments** for LLM training and evaluation. Agents solve PDDL planning problems using tool-based interaction in a sandboxed workspace.

---

## Features

- **Native PDDL support** -- load standard `.pddl` domain/problem files directly; 20 IPC benchmark domains included out of the box.
- **Agentic sandbox** -- miniSWE-style environment where an LLM agent uses coding tools (bash, read_file, write_file, str_replace) plus PDDL tools (validate, simulate) to solve planning problems.
- **Pure plan validation** -- standalone validator with step-by-step diagnostics, partial validation, and world state simulation.
- **Verifiers integration** -- native `MultiTurnEnv` for use with the [verifiers](https://github.com/primeintellect-ai/verifiers) framework and `prime eval`.
- **YAML-defined domains** -- for domains needing stochastic outcomes, energy management, or features beyond what PDDL expresses (kitchen, assembly).
- **World state engine** -- maintains objects, facts, numeric fluents, and enforces invariants.
- **Derived predicates** -- author higher-level semantics without extra code.
- **Numeric fluents, durations, energy** -- model time, resources, and stochastic outcomes.
- **Reward shaping & termination rules** -- milestones, potential-based shaping, and structured termination.
- **Multi-agent joint actions** -- one action per robot, applied simultaneously.
- **Reference plans & regression tests** -- validate domains and ensure backward compatibility.

---

## Architecture

```
src/sere/
+-- core/
|   +-- world_state.py       # Facts, objects, fluents, invariants
|   +-- semantics.py         # Clause + numeric evaluation, traces
|   +-- invariants.py        # Generic + domain-specific plugins
|   +-- validator.py         # Pure plan validation engine
|   +-- agentic_env.py       # miniSWE-style sandbox (tools + workspace)
|   +-- pddl_env/            # RL-style environment
|       +-- env.py           # Env: reset/step/reward/done
|       +-- engine.py        # Action application, stochastic outcomes
|       +-- planning.py      # Parse/execute action blocks
|       +-- rendering.py     # Messages + obs stitching
|       +-- prompt_formatter.py
|       +-- run_mode.py      # interactive / batch / open_loop
|
+-- pddl/
|   +-- pddl_parser.py       # Native PDDL domain/problem parser
|   +-- domain_spec.py       # DomainSpec, ActionSpec, PredicateSpec
|   +-- grounding.py         # S-expression grounding engine
|   +-- sexpr.py             # S-expression tokenizer/parser
|
+-- io/
|   +-- task_loader.py       # Unified loader (YAML + PDDL dispatch)
|   +-- pddl_loader.py       # PDDL directory loader + agentic task factory
|
+-- assets/
    +-- domain/              # YAML domains (kitchen, assembly)
    +-- tasks/               # Task YAMLs (per domain)
    +-- pddl/                # Native PDDL domains (20 IPC benchmarks)
        +-- blocksworld/     #   domain.pddl + extensions.yaml + problems/
        +-- logistics/
        +-- sokoban/
        +-- ...

integrations/
+-- verifiers/               # Verifiers framework integration
    +-- vf_sere.py           # SereEnv + AgenticSereEnv (MultiTurnEnv)
    +-- dataset.py           # Task discovery and dataset building
```

---

## Included Domains

### PDDL domains (20)

Standard IPC benchmarks, loaded natively from `.pddl` files. Each has 5-20 graded problem instances.

| Domain | Problems | Description |
|--------|----------|-------------|
| blocksworld | 15 | Stack blocks to match a goal configuration |
| miconic | 15 | Elevator: pick up and deliver passengers to floors |
| logistics | 15 | Move packages between cities via trucks and airplanes |
| gripper | 20 | Robot with two grippers moving balls between rooms |
| freecell | 18 | Card game: move cards to foundation piles |
| transport | 15 | Vehicles delivering packages across a road network |
| depots | 22 | Logistics with crates, hoists, and pallets |
| driverlog | 20 | Drivers walk/drive to deliver packages |
| zenotravel | 20 | Fly passengers between cities with fuel management |
| sokoban | 18 | Push boxes onto goal positions in a grid |
| barman | 18 | Bartender preparing cocktails with shakers |
| parking | 20 | Rearrange cars across curb/border positions |
| hiking | 18 | Couples hiking between waypoints with tents/cars |
| satellite | 20 | Point satellites and capture images |
| rovers | 20 | Mars rovers collecting and transmitting data |
| peg-solitaire | 22 | Jump pegs to clear the board |
| ferry | 10 | Ferry cars between locations |
| childsnack | 15 | Prepare and serve sandwiches to children |
| tpp | 15 | Travelling purchaser: buy and deliver goods |
| visitall | 15 | Visit every cell in a grid |

### YAML domains (2)

Extended domains with stochastic outcomes, energy management, and clutter generation.

| Domain | Tasks | Features |
|--------|-------|----------|
| kitchen | 15 | Stochastic actions, energy, numeric temps, derived predicates, multi-agent |
| assembly | 7 | Stochastic fastening, quality tracking, tool management, energy |

---

## Installation

```bash
git clone https://github.com/hallerite/SERE.git
cd SERE
uv sync
```

Requires **Python 3.11+**.

For the verifiers integration:

```bash
uv sync --extra verifiers
```

---

## Agentic Sandbox (Primary Mode)

The agentic sandbox gives an LLM a real filesystem workspace with PDDL files and coding tools. The agent reads the domain/problem, writes a plan, and validates it -- just like a coding agent solving a programming task.

### Workspace

```
workspace/
  domain.pddl   -- the planning domain (read-only)
  problem.pddl  -- the planning problem (read-only)
  plan.pddl     -- the agent writes its solution here
```

### Tools

| Tool | Description |
|------|-------------|
| `bash(command)` | Run shell commands in the workspace |
| `read_file(path)` | Read a file |
| `write_file(path, content)` | Create/overwrite a file |
| `str_replace(path, old_str, new_str)` | Targeted string replacement |
| `validate(up_to_step?)` | Validate plan.pddl (full plan = submission attempt) |
| `simulate(up_to_step?)` | Run plan.pddl and show resulting world state |

### Guards

- `domain.pddl` and `problem.pddl` are **read-only** (restored if modified via bash)
- Initial state is copied fresh for each validation (immutable)
- Goal expression is immutable
- Only full `validate()` counts as a submission attempt; partial validate and simulate are free

### Usage

```python
from sere.io.pddl_loader import load_agentic_task

env, meta = load_agentic_task("path/to/blocksworld", "path/to/instance-1.pddl")

# Get system prompt and tool schemas for the LLM
system_prompt = env.system_prompt()
tools = env.tool_schemas()

# Dispatch tool calls from the LLM
feedback, done = env.handle_tool_call("read_file", {"path": "domain.pddl"})
feedback, done = env.handle_tool_call("write_file", {
    "path": "plan.pddl",
    "content": "(pick-up A)\n(stack A B)\n",
})
feedback, done = env.handle_tool_call("validate", {})  # full submission

print(env.solved)    # True/False
print(env.attempts)  # number of full submissions
env.cleanup()        # remove temp workspace
```

---

## Verifiers Integration

SERE provides two `MultiTurnEnv` implementations for the verifiers framework:

- **`AgenticSereEnv`** -- agentic sandbox with tool use (recommended)
- **`SereEnv`** -- pure PDDL prompting (single-turn plan generation)

### With prime CLI

```bash
# Agentic (tool use)
prime eval run integrations.verifiers.vf_sere:load_agentic_environment \
  -m openai/gpt-4.1-mini -n 10 -r 1

# Pure PDDL prompting
prime eval run integrations.verifiers.vf_sere:load_environment \
  -m openai/gpt-4.1-mini -n 10
```

### Programmatic

```python
from integrations.verifiers import load_agentic_environment

env = load_agentic_environment(
    domains=["blocksworld", "gripper", "ferry"],
    num_tasks_per_domain=5,
    episodes_per_task=1,
    max_attempts=8,
)
```

See [integrations/verifiers/README.md](integrations/verifiers/README.md) for full details.

---

## Gym-Style API

For step-by-step RL-style interaction:

```python
from sere.io.task_loader import load_task

env, meta = load_task(
    "src/sere/assets/pddl/blocksworld",
    "src/sere/assets/pddl/blocksworld/problems/instance-1.pddl",
)

obs, info = env.reset()
obs, reward, done, info = env.step("(pick-up A)")
```

YAML tasks (kitchen, assembly):

```bash
uv run python -m sere.cli.run_task kitchen/t01_one_step_steep.yaml
```

---

## PDDL Domain Structure

Each PDDL domain lives in `assets/pddl/<name>/`:

```
blocksworld/
+-- domain.pddl         # Standard PDDL domain definition
+-- extensions.yaml     # Optional: reward shaping, metadata
+-- problems/
    +-- instance-1.pddl # Small (4 blocks)
    +-- instance-2.pddl
    +-- ...             # Graded by difficulty
```

---

## Tests

```bash
uv run python -m pytest tests/ -q
```

438 tests, 0 failures.
