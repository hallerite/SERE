# SERE (Verifiers Environment)

SERE is a symbolic-first embodied environment for controllable RL with LLMs.
It exposes typed objects, predicates, numeric fluents, and actions with explicit
preconditions/effects. Instead of pixels, the agent interacts through text
observations and PDDL-style actions. This makes runs fully interpretable and
reproducible while letting you control horizon length, feedback frequency, and
observability.

More background and full docs live in the SERE repo:
https://github.com/hallerite/SERE

## Quick Start

```python
from vf_sere import load_environment

# All tasks across all domains (YAML + PDDL)
env = load_environment()

# Only YAML domains
env = load_environment(domains=["kitchen", "assembly"])

# Only PDDL domains
env = load_environment(domains=["blocksworld", "logistics", "sokoban"])

# Mix both
env = load_environment(domains=["kitchen", "blocksworld", "gripper"])
```

Prime CLI (uses the env name, no imports needed):

```bash
prime eval run vf-sere -m gpt-4o-mini -n 10
```

## Available Domains

### PDDL domains (20)

Standard IPC benchmarks loaded from native `.pddl` files with 5-20 problem
instances each: `blocksworld`, `miconic`, `logistics`, `gripper`, `freecell`,
`transport`, `depots`, `driverlog`, `zenotravel`, `sokoban`, `barman`,
`parking`, `hiking`, `satellite`, `rovers`, `peg-solitaire`, `ferry`,
`childsnack`, `tpp`, `visitall`.

### YAML domains (2)

Extended domains with stochastic outcomes, energy, and NL templates:
`kitchen` (15 tasks), `assembly` (7 tasks).

### Listing domains programmatically

```python
from vf_sere.dataset import get_available_domains
print(get_available_domains())  # {'kitchen', 'assembly', 'blocksworld', ...}
```

## Key Knobs (Controllability)

Use these to shape difficulty, horizon, and feedback:

- `run_mode`: `INTERACTIVE` (step-by-step), `BATCH` (short plan segments),
  `OPEN_LOOP` (full plan, no intermediate feedback). This directly controls
  feedback frequency and effective horizon.
- `max_episode_steps`: override per-task step limits.
- `time_limit`: cap total time when durations are enabled.
- `enable_durations`: actions consume time; enables time-based horizons.
- `enable_numeric`: numeric fluents (energy, temperature, etc.).
- `enable_stochastic`: stochastic outcomes for robustness testing.
- `enable_reward_shaping`: apply task-defined reward shaping milestones (default: False).
- `reward_shaping`: milestone or potential-based shaping override (task-level control).
- `step_penalty` / `invalid_penalty`: trade off exploration vs. efficiency.

Observation control (partial observability):

- `show_domain_pddl`: include raw PDDL domain in system prompt (default: True).
- `show_affordances`: list currently valid actions (disabled by default for PDDL domains to avoid expensive grounding).
- `formatter_config`: set `visibility` (e.g., `room`) to restrict what the
  agent can see.

## Task Selection

```python
env = load_environment(
    domains=["kitchen", "blocksworld"],
    num_tasks_per_domain=5,
    include_multi_agent=True,
    include_pddl=True,      # include PDDL domains (default: True)
    episodes_per_task=2,
)
```

Tasks are discovered automatically from both `sere.assets.tasks` (YAML) and
`sere.assets.pddl` (PDDL problem files).

## Multi-Agent Tasks

Multi-agent tasks (currently kitchen only) require one action per robot per step.
The parser supports multiple S-expressions in a single response:

```
(move r1 kitchen pantry)(idle r2)
```

## Example: More Control

```python
from vf_sere import load_environment
from sere.core.pddl_env.run_mode import RunMode

env = load_environment(
    domains=["kitchen"],
    run_mode=RunMode.BATCH,
    max_episode_steps=40,
    enable_durations=True,
    time_limit=30.0,
    enable_stochastic=False,
    show_affordances=False,
)
```

## Troubleshooting

- Import error: install SERE with verifiers extra:
  `uv sync --extra verifiers`
- No tasks found: check available domains via `get_available_domains()`.
- PDDL domain timeout: ensure `show_affordances=False` (default for PDDL domains) to skip expensive action enumeration.
