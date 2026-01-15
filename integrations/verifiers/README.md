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

# All tasks across all domains
env = load_environment()

# Or limit to specific domains
env = load_environment(domains=["kitchen", "assembly"])
```

Prime CLI (uses the env name, no imports needed):

```bash
prime eval run vf-sere -m gpt-4o-mini -n 10
```

Suggested default eval scope (first 5 kitchen + first 5 assembly tasks):

```bash
prime eval run vf-sere -m gpt-4o-mini -n 20 \
  -a '{"domains":["kitchen","assembly"],"num_tasks_per_domain":5}'
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
- `reward_shaping`: milestone or potential-based shaping (task-level control).
- `step_penalty` / `invalid_penalty`: trade off exploration vs. efficiency.

Observation control (partial observability):

- `display_nl`: include natural-language glosses for facts/actions.
- `show_affordances`: list currently valid actions.
- `formatter_config`: set `visibility` (e.g., `room`) to restrict what the
  agent can see.

## Task Selection

```python
env = load_environment(
    domains=["kitchen"],
    num_tasks_per_domain=5,
    include_multi_agent=True,
    episodes_per_task=2,
)
```

Tasks are YAML-defined (domain + task). SERE ships multiple domains (e.g.,
`kitchen`, `assembly`) with reference plans for solvability checks.

## Multi-Agent Tasks

Multi-agent tasks require one action per robot per step. The parser supports
multiple S-expressions in a single response:

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
    display_nl=False,
    show_affordances=False,
)
```

## Troubleshooting

- Import error: install SERE with verifiers extra:
  `uv sync --extra verifiers`
- No tasks found: check available domains via `get_available_domains()`.
