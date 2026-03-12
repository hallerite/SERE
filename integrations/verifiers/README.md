# SERE (Verifiers Integration)

Verifiers integration for SERE -- symbolic planning environments for LLM evaluation and training.

Two environment modes:

- **Agentic** (`AgenticSereEnv`) -- miniSWE-style sandbox with tool use. The agent reads domain/problem files, writes a plan, and validates it using coding tools. Recommended for evaluation and RL training.
- **Pure PDDL** (`SereEnv`) -- single-turn plan generation. The agent receives the domain + problem as a prompt and outputs a plan directly.

## Quick Start

### Agentic (tool use)

```python
from integrations.verifiers import load_agentic_environment

env = load_agentic_environment(
    domains=["blocksworld", "gripper", "ferry"],
    num_tasks_per_domain=5,
    episodes_per_task=1,
    max_attempts=8,
)
```

### Pure PDDL

```python
from integrations.verifiers import load_environment

env = load_environment(
    domains=["blocksworld", "logistics", "sokoban"],
    num_tasks_per_domain=5,
)
```

### Prime CLI

```bash
# Agentic
prime eval run integrations.verifiers.vf_sere:load_agentic_environment \
  -m openai/gpt-4.1-mini -n 10 -r 1

# Pure PDDL
prime eval run integrations.verifiers.vf_sere:load_environment \
  -m openai/gpt-4.1-mini -n 10
```

## Agentic Environment

The agentic environment creates a sandboxed workspace per episode with:

```
workspace/
  domain.pddl   (read-only)
  problem.pddl  (read-only)
  plan.pddl     (agent writes solution here)
```

### Tools

| Tool | Description |
|------|-------------|
| `bash(command)` | Run shell commands in the workspace |
| `read_file(path)` | Read a file |
| `write_file(path, content)` | Create/overwrite a file |
| `str_replace(path, old_str, new_str)` | Targeted string replacement |
| `validate(up_to_step?)` | Validate plan.pddl against domain+problem |
| `simulate(up_to_step?)` | Run plan and show resulting world state |

Guards:
- `domain.pddl` and `problem.pddl` are read-only (restored if modified)
- Only full `validate()` (no `up_to_step`, or `up_to_step >= total`) counts as a submission attempt
- `max_attempts` (default 8) limits full submission attempts per episode

### Configuration

```python
env = load_agentic_environment(
    domains=["blocksworld"],       # filter to specific domains
    num_tasks_per_domain=10,       # limit problems per domain
    episodes_per_task=1,           # rollouts per problem
    eval_episodes_per_task=3,      # rollouts per problem for eval
    max_attempts=8,                # max full validation attempts
    enable_numeric=False,          # numeric fluent support
    enable_conditional=False,      # conditional effect support
    seed=0,                        # dataset shuffle seed
)
```

## Pure PDDL Environment

The pure PDDL environment gives the agent the domain definition as a system prompt and the problem as a user message. The agent responds with a plan (one action per line).

### Configuration

```python
env = load_environment(
    domains=["kitchen", "blocksworld"],
    num_tasks_per_domain=5,
    include_multi_agent=True,
    include_pddl=True,
    include_yaml=True,
    episodes_per_task=2,
    run_mode=RunMode.OPEN_LOOP,       # INTERACTIVE, BATCH, or OPEN_LOOP
    max_episode_steps=50,
    show_domain_pddl=True,            # include raw PDDL domain in prompt
    show_affordances=False,            # list valid actions (expensive)
    enable_durations=False,
    enable_numeric=False,
    enable_stochastic=False,
    enable_reward_shaping=False,
)
```

## Available Domains

### PDDL domains (20)

Standard IPC benchmarks with 5-20 problem instances each: `blocksworld`, `miconic`, `logistics`, `gripper`, `freecell`, `transport`, `depots`, `driverlog`, `zenotravel`, `sokoban`, `barman`, `parking`, `hiking`, `satellite`, `rovers`, `peg-solitaire`, `ferry`, `childsnack`, `tpp`, `visitall`.

### YAML domains (2)

Extended domains with stochastic outcomes and energy: `kitchen` (15 tasks), `assembly` (7 tasks).

### Listing domains

```python
from integrations.verifiers.dataset import get_available_domains
print(get_available_domains())  # {'kitchen', 'assembly', 'blocksworld', ...}
```

## Troubleshooting

- **Import error**: install SERE with verifiers extra: `uv sync --extra verifiers`
- **No tasks found**: check available domains via `get_available_domains()`
- **PDDL domain timeout**: ensure `show_affordances=False` (default) to skip expensive action enumeration
