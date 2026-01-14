# SERE Verifiers Integration

Train LLM agents on SERE symbolic reasoning tasks using the [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) RL framework.

## Installation

```bash
# Install SERE with Verifiers support
uv sync --extra verifiers

# Or with pip
pip install -e ".[verifiers]"
```

## Quick Start

```python
from integrations.verifiers import load_environment

# Load all SERE tasks from all domains
env = load_environment()

# Or load specific domains
env = load_environment(domains=["kitchen", "assembly"])

# Or load specific tasks
env = load_environment(
    task_paths=[
        "kitchen/t01_one_step_steep.yaml",
        "kitchen/t02_two_step_brew.yaml",
    ]
)

# Evaluate with a model
results = env.evaluate(
    client=client,
    model="gpt-4o-mini",
    num_rollouts=10,
)

print(f"Average reward: {sum(results['reward']) / len(results['reward'])}")
```

## Architecture

This integration uses Verifiers' experimental **GymEnv** class, which provides a universal wrapper for Gym-compatible environments:

```
┌─────────────────────────────────────┐
│     Verifiers GymEnv                │
│  (Multi-turn RL environment)        │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│     SereGymWrapper                   │
│  • reset(seed) → loads task[seed]   │
│  • step(action) → PDDL execution    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│     SERE PDDLEnv                     │
│  (Symbolic reasoning environment)   │
└─────────────────────────────────────┘
```

### Key Components

1. **SereGymWrapper** (`wrapper.py`)
   - Implements Gym protocol (reset/step) for SERE tasks
   - Maps seed to task index for multi-task training
   - Passes through SERE's reward signals

2. **Parser** (`parser.py`)
   - Extracts PDDL S-expressions from model outputs
   - Supports multi-agent tasks (multiple actions per turn)
   - Handles various formats (raw, explained, embedded)

3. **Dataset Discovery** (`dataset.py`)
   - Auto-discovers SERE tasks from `assets/tasks/`
   - Filters by domain, multi-agent, etc.

4. **Load Environment** (`__init__.py`)
   - Main entry point: `load_environment()`
   - Configures GymEnv with SERE wrapper
   - Returns ready-to-train environment

## Configuration

### Task Selection

```python
# All tasks from all domains (default)
env = load_environment()

# Specific domains
env = load_environment(domains=["kitchen", "assembly"])

# Limit tasks per domain
env = load_environment(
    domains=["kitchen"],
    num_tasks_per_domain=5
)

# Exclude multi-agent tasks
env = load_environment(include_multi_agent=False)

# Explicit task list
env = load_environment(
    task_paths=[
        "kitchen/t01_one_step_steep.yaml",
        "assembly/t01_one_step_fasten.yaml",
    ]
)
```

### Episode Configuration

Control how training episodes are distributed across tasks:

```python
env = load_environment(
    domains=["kitchen"],
    num_tasks_per_domain=5,        # Load 5 kitchen tasks
    episodes_per_task=3,           # Train 3 episodes per task
    eval_episodes_per_task=2,      # Evaluate 2 episodes per task
    max_episode_steps=50,          # Override task max_steps
    seed=42,                       # Random seed
)
# Result: 15 training episodes (5 tasks × 3 episodes)
# Episode 0: task 0 (1st time), Episode 1: task 1 (1st time), ...
# Episode 5: task 0 (2nd time), Episode 6: task 1 (2nd time), ...
```

**Multi-task training**: With multiple tasks, the agent cycles through them using `seed % num_tasks` to select which task to load. This provides curriculum learning across diverse scenarios rather than overfitting to a single task.

### Environment Features

Toggle PDDL features to control environment complexity:

#### `enable_numeric` - Numeric fluents (default: True)

When enabled, the environment includes numeric fluents like battery levels, temperatures, and quantities:
```python
env = load_environment(enable_numeric=True)
# Observations include: (= (energy robot) 8.00), (= (water-temp tumbler) 100.00)
# Actions can modify numeric values: (recharge robot charger) increases battery
```

When disabled, only boolean predicates are used (classical PDDL):
```python
env = load_environment(enable_numeric=False)
# Only boolean facts: (at robot kitchen), (holding robot cup)
```

#### `enable_conditional` - Conditional effects (default: True)

When enabled, actions can have conditional side effects:
```python
env = load_environment(enable_conditional=True)
# Example: (heat stove pot) heats the pot AND heats any water inside it
```

When disabled, each action has only its direct effects.

#### `enable_durations` - Time-based actions (default: True)

When enabled, actions have durations and consume time:
```python
env = load_environment(enable_durations=True)
# Each action advances time: (heat stove) takes 5.0 time units
# Observations show: Time: 5.0/30.0
```

When disabled, all actions are instantaneous (classical PDDL).

#### `enable_stochastic` - Probabilistic outcomes (default: False)

When enabled, actions may fail or have probabilistic effects:
```python
env = load_environment(enable_stochastic=True)
# Actions may fail: (pick-up robot cup) might drop the cup
# Same action can produce different outcomes across episodes
```

When disabled, actions are deterministic.

### Reward Configuration

Control reward structure for training:

#### `step_penalty` - Reward per action (default: -0.01)

Each action incurs this reward (typically negative to encourage efficiency):
```python
env = load_environment(step_penalty=-0.01)
# Each action costs -0.01 reward, so a 10-step solution gets -0.1 from steps

env = load_environment(step_penalty=-0.05)
# Higher penalty: agent is penalized more for longer solutions

env = load_environment(step_penalty=-0.001)
# Lower penalty: agent is less penalized for exploration
```

#### `invalid_penalty` - Penalty for invalid actions (default: -0.1)

When the agent attempts an invalid action (e.g., picking up an object that's not present):
```python
env = load_environment(invalid_penalty=-0.1)
# Invalid action costs -0.1 reward and episode continues

env = load_environment(invalid_penalty=-0.5)
# Higher penalty discourages invalid actions more strongly
```

#### `time_limit` - Maximum time in time units (default: None)

When set with `enable_durations=True`, episode terminates if time exceeds limit:
```python
env = load_environment(
    enable_durations=True,
    time_limit=30.0
)
# Episode fails if total action durations exceed 30.0 time units
```

#### `reward_shaping` - Intermediate rewards

**Milestone-based**: Reward when specific conditions are achieved
```python
env = load_environment(
    reward_shaping={
        "mode": "milestone",
        "milestones": [
            {"expr": "(has-hot-water ?c)", "reward": 0.3, "once": True},
            {"expr": "(in ?bag ?c)", "reward": 0.2, "once": True},
        ]
    }
)
# Agent receives +0.3 when water is heated (once per episode)
# Agent receives +0.2 when tea bag is added (once per episode)
# Final success still gives +1.0, so total can be up to +1.5
```

**Potential-based**: Reward based on progress toward goal
```python
env = load_environment(
    reward_shaping={
        "mode": "potential",
        "gamma": 0.99,
    }
)
# Reward at each step based on estimated distance to goal
# Provides dense learning signal throughout episode
```

### Observation Control

Control what information appears in observations shown to the LLM:

#### `display_nl` - Natural language descriptions (default: True)

When `display_nl=True`, each PDDL predicate includes an English explanation:
```
State:
- (at robot kitchen) – robot is at kitchen
- (has-hot-water tumbler) – tumbler contains hot water

Valid moves:
- (steep-tea robot sachet tumbler) – Steep tea from sachet in tumbler
```

When `display_nl=False`, only PDDL syntax is shown:
```
State:
- (at robot kitchen)
- (has-hot-water tumbler)
```

#### `show_affordances` - Available actions list (default: True)

When `show_affordances=True`, observations include a "Valid moves" section listing all currently applicable actions:
```
Valid moves:
- (pick-up robot tumbler) – Pick up tumbler
- (pick-up robot kettle) – Pick up kettle
- (steep-tea robot sachet tumbler) – Steep tea from sachet in tumbler
```

When `show_affordances=False`, this section is omitted. The agent must infer which actions are valid from the current state.

#### Example: Full vs. Minimal Observations

**With full observations (default):**
```python
env = load_environment(
    domains=["kitchen"],
    display_nl=True,         # Natural language explanations
    show_affordances=True,   # List valid actions
)
```
**Observation shown to LLM:**
```
Steps: 0/3 | Time: 0.00/4.00 | Energy: robot 8.00/8.00

State:
- (at robot kitchen) – robot is at kitchen
- (clear-hand robot) – robot has a free hand
- (has-hot-water tumbler) – tumbler contains hot water
- (in sachet tumbler) – sachet is in tumbler
- (obj-at kettle kitchen) – kettle is at kitchen
- (obj-at tumbler kitchen) – tumbler is at kitchen

Goal:
- (tea-ready tumbler) – Tea is ready in tumbler

Fluents:
- (= (battery-cap robot) 8.00)
- (= (energy robot) 8.00)
- (= (water-temp tumbler) 100.00)

Valid moves:
- (pick-up robot tumbler) – Pick up tumbler
- (steep-tea robot sachet tumbler) – Steep tea from sachet in tumbler
```

**With minimal observations:**
```python
env = load_environment(
    domains=["kitchen"],
    display_nl=False,        # PDDL only, no natural language
    show_affordances=False,  # No valid actions list
)
```
**Observation shown to LLM:**
```
Steps: 0/3 | Time: 0.00/4.00 | Energy: robot 8.00/8.00

State:
- (at robot kitchen)
- (clear-hand robot)
- (has-hot-water tumbler)
- (in sachet tumbler)
- (obj-at kettle kitchen)
- (obj-at tumbler kitchen)

Goal:
- (tea-ready tumbler)

Fluents:
- (= (battery-cap robot) 8.00)
- (= (energy robot) 8.00)
- (= (water-temp tumbler) 100.00)

What will be your next move?
```

Note: Goal, fluents (when `enable_numeric=True`), and action feedback messages are always shown.

### Custom System Prompt

```python
env = load_environment(
    system_prompt="""
    You are a robot solving planning tasks.
    Think step by step and use PDDL actions.
    Format: (action-name arg1 arg2 ...)
    """
)
```

## Multi-Agent Tasks

SERE's multi-agent tasks require coordinating multiple robots. This integration uses a "flattened" approach where the agent controls all robots simultaneously:

```python
# Multi-agent task with r1, r2, r3
# Agent provides multiple actions in one response:
Assistant: """
I'll coordinate the robots:
(move r1 kitchen pantry)
(pick-up r2 leaf)
(idle r3)
"""
```

The parser automatically extracts multiple S-expressions and passes them to SERE's joint action execution.

## Rewards

By default, uses SERE's built-in rewards:
- **Step penalty**: -0.01 per action (configurable in task YAML)
- **Invalid action penalty**: -0.1 for invalid moves
- **Terminal reward**: +1.0 for success (configurable)
- **Optional shaping**: Milestone or potential-based (if configured)

The `GymEnv` automatically sums step rewards across the episode.

## Examples

### Basic Evaluation

```python
from integrations.verifiers import load_environment
from openai import AsyncOpenAI

env = load_environment(domains=["kitchen"], num_tasks_per_domain=5)

client = AsyncOpenAI(base_url="http://localhost:8000/v1")

results = env.evaluate(
    client=client,
    model="gpt-4o-mini",
    num_rollouts=10,
)

print(f"Success rate: {sum(r > 0 for r in results['reward']) / len(results['reward'])}")
```

### Custom Rubric

```python
import verifiers as vf
from integrations.verifiers import load_environment, SereGymWrapper, parse_pddl_actions

# Build custom rubric
async def efficiency_bonus(state: vf.State) -> float:
    num_steps = len(state["trajectory"])
    max_steps = state["info"]["max_steps"]
    return max(0.0, 1.0 - num_steps / max_steps)

rubric = vf.Rubric(
    funcs=[vf.envs.experimental.gym_env.sum_step_rewards, efficiency_bonus],
    weights=[1.0, 0.2],
)

# Create environment with custom rubric
env = vf.envs.experimental.gym_env.GymEnv(
    env_cls=SereGymWrapper,
    env_kwargs={
        "task_paths": ["kitchen/t01_one_step_steep.yaml"],
        "env_kwargs": {},
    },
    action_parser=parse_pddl_actions,
    obs_to_text=lambda x: x,
    num_train_episodes=100,
    num_eval_episodes=20,
    rubric=rubric,
    system_prompt="Custom prompt here...",
)
```

## Available Tasks

Discover available domains and tasks:

```python
from integrations.verifiers import get_available_domains, discover_tasks

# List all domains
domains = get_available_domains()
print(f"Available domains: {domains}")

# Discover tasks in a domain
kitchen_tasks = discover_tasks(domains=["kitchen"])
print(f"Kitchen tasks: {kitchen_tasks}")

# Find multi-agent tasks only
multi_agent_tasks = discover_tasks(include_multi_agent=True)
multi_agent_tasks = [t for t in multi_agent_tasks if "multi_agent" in t]
```

## Integration with Prime Environments

This integration is also available as a Prime Environments package:

```bash
# Install from Prime Hub
prime env install primeintellect/sere

# Use in training config
prime eval run sere -m gpt-4o-mini -n 100
```

See the [Prime Hub package README](../../../../prime-environments/environments/sere/README.md) for more details.

## Troubleshooting

### Import Error

```
ImportError: Verifiers integration requires verifiers
```

**Solution**: Install with verifiers extra:
```bash
uv sync --extra verifiers
```

### No Tasks Found

```
ValueError: No tasks found. Domains: ['foo'], include_multi_agent: True
```

**Solution**: Check available domains:
```python
from integrations.verifiers import get_available_domains
print(get_available_domains())
```

### Action Parsing Error

If the model produces invalid PDDL:
- Check system prompt guides the format
- Review model outputs in failed rollouts
- Consider using examples in few-shot

## Development

Run tests:
```bash
pytest tests/test_verifiers_integration.py -v
```

## Architecture Details

### Why GymEnv?

SERE's `PDDLEnv` already implements the Gym interface (`reset()`, `step()`). Verifiers' `GymEnv` provides:
- Automatic dataset generation from environment resets
- Trajectory management with rewards
- Parser integration
- Cleanup lifecycle management

This is simpler than implementing `MultiTurnEnv` from scratch and more natural for episodic environments.

### Reward Accumulation

GymEnv uses `EpisodicSumRubric` by default, which sums rewards across trajectory steps:
```python
total_reward = sum(step["reward"] for step in state["trajectory"])
```

SERE provides per-step rewards, which are stored in each trajectory step and summed at the end.

### Task-to-Dataset Mapping

Each dataset row represents one task:
- `question`: Initial observation from `env.reset()`
- `answer`: Seed/index (used to select which task to load)
- Row count = `num_tasks × episodes_per_task`

During rollout:
1. Extract seed from `state["answer"]`
2. Use seed to select task: `task_idx = seed % len(task_paths)`
3. Load and reset that task
4. Execute episode

This allows multiple episodes per task for better sample efficiency.