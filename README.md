# Symbolic Embodied Reasoning Environments (SERE)

**SERE** is a lightweight framework for building **symbolic, embodied reasoning environments** — where agents must manipulate objects, respect spatial and causal constraints, and satisfy task goals expressed in PDDL-style logic.

It's designed for **RL + LLM training**, giving you a **Gym-style API** but with symbolic state, grounded actions, stochasticity, and reward shaping built in.

---

## Features

- **Native PDDL support** – load standard `.pddl` domain/problem files directly; 20 IPC benchmark domains included out of the box.
- **YAML-defined domains** – for domains that need stochastic outcomes, NL glosses, or energy management beyond what PDDL expresses (kitchen, assembly).
- **PDDL-style grounding** – parses `(pick-up r1 mug1)` into concrete state updates.
- **World state engine** – maintains objects, facts, numeric fluents, and enforces invariants.
- **Derived predicates** – author higher-level semantics without extra code.
- **Numeric fluents, durations, energy** – model time, resources, and stochastic outcomes.
- **Reward shaping & termination rules** – instant milestones, potential-based shaping, and structured `all/any` termination.
- **Multi-agent joint actions** – require one action per robot and apply effects simultaneously.
- **Invariant plugins** – register domain-specific constraints (e.g. "object can't be in two places").
- **Human-readable rendering** – natural language + PDDL observations for LLM prompting, with optional affordance lists.
- **Reference plans & regression tests** – validate domains and ensure backward compatibility.

---

## Architecture

```
src/sere/
├── core/
│   ├── world_state.py       # Facts, objects, fluents, invariants
│   ├── semantics.py         # Clause + numeric evaluation, traces
│   ├── invariants.py        # Generic + domain-specific plugins
│   └── pddl_env/            # RL-style environment + prompting
│       ├── env.py           # Env: reset/step/reward/done
│       ├── engine.py        # Action application, stochastic outcomes
│       ├── planning.py      # Parse/execute action blocks
│       ├── rendering.py     # Messages + obs stitching
│       ├── prompt_formatter.py # System prompt + observations + affordances
│       └── run_mode.py      # interactive / batch / open_loop
│
├── pddl/
│   ├── pddl_parser.py       # Native PDDL domain/problem parser
│   ├── domain_spec.py       # DomainSpec, ActionSpec, PredicateSpec, …
│   ├── grounding.py         # S-expression grounding engine
│   └── sexpr.py             # S-expression tokenizer/parser
│
├── io/
│   ├── task_loader.py       # Unified loader (YAML + PDDL dispatch)
│   └── pddl_loader.py       # PDDL directory loader
│
├── cli/
│   └── run_task.py
│
└── assets/
    ├── domain/              # YAML domains (kitchen, assembly)
    ├── tasks/               # Task YAMLs (per domain)
    └── pddl/                # Native PDDL domains (20 IPC benchmarks)
        ├── blocksworld/     #   domain.pddl + extensions.yaml + problems/
        ├── logistics/
        ├── sokoban/
        └── …
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

Extended domains with stochastic outcomes, energy management, NL templates, and clutter generation.

| Domain | Tasks | Features |
|--------|-------|----------|
| kitchen | 15 | Stochastic actions (5-10% failure), energy, numeric temps, derived predicates, multi-agent, clutter |
| assembly | 7 | Stochastic fastening (40% defect rate), quality tracking, tool management, energy |

---

## Installation

```bash
git clone https://github.com/hallerite/SERE.git
cd SERE
uv sync
```

Requires **Python 3.11+**.

---

## Running a Task

YAML tasks (kitchen, assembly):

```bash
uv run python -m sere.cli.run_task kitchen/t01_one_step_steep.yaml
```

PDDL tasks can be loaded programmatically:

```python
from sere.io.task_loader import load_task

# YAML task
env, meta = load_task(None, "kitchen/t01_one_step_steep.yaml")

# PDDL task (pass path to .pddl problem file)
env, meta = load_task(
    "src/sere/assets/pddl/blocksworld",
    "src/sere/assets/pddl/blocksworld/problems/instance-1.pddl",
)

obs, info = env.reset()
obs, reward, done, info = env.step("(pick-up A)")
```

---

## PDDL Domain Structure

Each PDDL domain lives in `assets/pddl/<name>/`:

```
blocksworld/
├── domain.pddl         # Standard PDDL domain definition
├── extensions.yaml     # Optional: NL templates, reward shaping, metadata
└── problems/
    ├── instance-1.pddl # Small (4 blocks)
    ├── instance-2.pddl
    └── …               # Graded by difficulty
```

The `extensions.yaml` overlay is optional and provides NL glosses, outcome descriptions, and reward shaping hints that standard PDDL can't express. If omitted, NL templates are auto-generated from predicate/action names.

Affordance listing (enumerating all valid actions) is disabled by default for PDDL domains to avoid expensive grounding on large state spaces. The LLM must infer valid actions from the state and action schemas.

---

## Ludic integration (multi-agent)

SERE includes a Ludic wrapper that exposes each robot as a separate Ludic agent.
Agent IDs are the sorted robot symbols (e.g. `r1`, `r2`). Actions are **raw PDDL
S-expressions** with exactly one action per agent per step. Use `(idle r)` for no-op.

```python
from integrations.ludic import SereLudicEnv, pddl_action_parser
from sere.core.pddl_env.run_mode import RunMode
from sere.io.task_loader import load_task

env, meta = load_task(
    None,
    "kitchen/t11_multi_agent_parallel_brew.yaml",
    run_mode=RunMode.INTERACTIVE,
)
ludic_env = SereLudicEnv(env)

parser = pddl_action_parser()
# agent_map = {aid: Agent(..., parser=parser, ...) for aid in ludic_env.agent_ids}
# protocol = MultiAgentProtocol(agent_map)
```

---

## Authoring Domains & Tasks

### PDDL domains (preferred for new domains)

Write a standard `domain.pddl` and problem files. Optionally add `extensions.yaml` for NL templates and reward shaping. See any domain in `assets/pddl/` for examples.

### YAML domains (for stochastic/extended features)

- **Domains** (`assets/domain/*.yaml`) define types, predicates, fluents, actions (with stochastic outcomes, conditional + numeric effects, durations), and derived predicates.

- **Tasks** (`assets/tasks/**/*.yaml`) define objects, initial state, statics, termination rules, optional reward shaping, and reference plans.

This separation makes it easy to randomize tasks or auto-generate curricula.

## To Do
- Migrate simple YAML domains (floortile, goldminer, openstacks, rovers, satellite) to native PDDL
- Extend `extensions.yaml` schema to support stochastic outcomes, enabling kitchen/assembly migration
- Add task/domain randomization hooks
- Add more domain clusters and cross-domain skill tags
