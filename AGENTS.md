# Agent Architecture in SERE

This document describes how agents work in SERE (Symbolic Embodied Reasoning Environments) and how to use them in both single-agent and multi-agent scenarios.

---

## Table of Contents

1. [Overview](#overview)
2. [What are Agents in SERE?](#what-are-agents-in-sere)
3. [Single-Agent vs Multi-Agent](#single-agent-vs-multi-agent)
4. [Agent-Environment Interface](#agent-environment-interface)
5. [Ludic Integration](#ludic-integration)
6. [Creating and Using Agents](#creating-and-using-agents)
7. [Agent Action Format](#agent-action-format)
8. [Examples](#examples)
9. [Best Practices](#best-practices)

---

## Overview

SERE provides a **symbolic embodied reasoning environment** where agents must manipulate objects, respect spatial and causal constraints, and satisfy task goals expressed in PDDL-style logic. Agents in SERE are entities (typically robots) that:

- Perceive the world state through observations
- Reason about actions and their effects
- Execute grounded actions (e.g., `(move r1 kitchen pantry)`)
- Collaborate with other agents in multi-agent scenarios

SERE supports both **single-agent** and **multi-agent** modes, with seamless integration into the **Ludic** framework for LLM-based agent training and evaluation.

---

## What are Agents in SERE?

In SERE, an **agent** is any entity of type `robot` defined in the domain. Each robot is an autonomous actor that can:

- Navigate between locations (`move`)
- Manipulate objects (`pick-up`, `put-down`, `put-in`)
- Operate appliances (`power-on`, `heat-kettle`)
- Coordinate with other robots (in multi-agent mode)

### Agent Identity

Agents are identified by their **symbol** in the task definition. For example:

```yaml
objects:
  r1: robot
  r2: robot
```

Here, `r1` and `r2` are two distinct agents. In multi-agent mode, each robot becomes a separate Ludic agent with its own observation, action, and reward stream.

### Agent Properties

Each agent may have:

- **Position**: Where the agent is located (`(at r1 kitchen)`)
- **Hand state**: What the agent is holding (`(holding r1 mug1)` or `(clear-hand r1)`)
- **Energy**: Numeric fluent tracking battery level (`(energy r1)`)
- **Actions**: Domain-specific capabilities defined in the domain YAML

---

## Single-Agent vs Multi-Agent

### Single-Agent Mode (default)

In single-agent mode:
- There is typically **one robot** in the environment
- The robot executes actions sequentially
- Each `step()` call expects **exactly one action**

Example task (single-agent):
```yaml
meta:
  multi_agent: false  # or omit this field (default)

objects:
  r1: robot
  # ... other objects
```

### Multi-Agent Mode

In multi-agent mode:
- Multiple robots operate **simultaneously**
- Each robot must provide **exactly one action per step** (or use `(idle r)` for no-op)
- Actions are executed as a **joint action** with simultaneous effects
- Missing actions from any robot result in an `invalid_move` outcome

Example task (multi-agent):
```yaml
meta:
  multi_agent: true

objects:
  r1: robot
  r2: robot
  # ... other objects
```

#### Joint Action Execution

In multi-agent mode, SERE uses `execute_joint()` which:

1. **Validates**: Ensures exactly one action per robot
2. **Applies**: Executes all robot actions simultaneously
3. **Updates**: Applies all add/delete effects atomically
4. **Checks**: Validates invariants and termination conditions

This allows for true **parallel execution** where robots can work on complementary tasks simultaneously.

---

## Agent-Environment Interface

### PDDLEnv

The core environment class is `PDDLEnv` (see `src/sere/core/pddl_env/env.py`). It provides:

```python
class PDDLEnv:
    def reset(self, *, seed: Optional[int] = None) -> Tuple[str, dict]:
        """Reset the environment and return initial observation + info."""

    def step(self, text: str) -> Tuple[str, float, bool, dict]:
        """Execute action(s) and return (obs, reward, done, info)."""
```

#### Observation Format

Observations include:
- **State**: Current facts (e.g., `(at r1 kitchen)`, `(clear-hand r1)`)
- **Goal**: Task objectives (e.g., `(tea-ready mug1)`)
- **Fluents**: Numeric state (e.g., `energy(r1) = 8.5`, `time = 12.0`)
- **Affordances**: Available actions for each robot
- **Messages**: Feedback from previous actions

Example observation:
```
State:
  - (at r1 kitchen) – r1 is at kitchen
  - (obj-at kettle1 kitchen) – kettle1 is at kitchen
  - (clear-hand r1) – r1 has a free hand

Fluents:
  - energy(r1) = 10.0
  - time = 0.0

Goal:
  - (tea-ready mug1)

Affordances for r1:
  - (move r1 kitchen <l>): Move to an adjacent location
  - (pick-up r1 kettle1): Pick up an object
  - (open r1 kettle1): Open a container
```

#### Action Format

Actions are PDDL S-expressions:
```
(action-name robot-id arg1 arg2 ...)
```

Examples:
- `(move r1 kitchen pantry)` – Move r1 from kitchen to pantry
- `(pick-up r1 mug1)` – r1 picks up mug1
- `(heat-kettle r1 kettle1 4)` – r1 heats kettle1 for 4 time units
- `(idle r1)` – r1 does nothing (no-op)

### Run Modes

SERE supports three run modes (see `src/sere/core/pddl_env/run_mode.py`):

1. **`INTERACTIVE`** (default): Exactly one action per step
   - Single-agent: 1 action total
   - Multi-agent: 1 action per robot

2. **`BATCH`**: Multiple sequential actions per step (plan execution)
   - Executes a sequence of actions atomically
   - Useful for pre-planned trajectories

3. **`OPEN_LOOP`**: Execute entire plan without intermediate observations
   - Fast execution for testing reference plans

---

## Ludic Integration

SERE provides a **Ludic-compatible wrapper** (`SereLudicEnv`) that exposes each robot as a separate Ludic agent. This enables seamless integration with LLM-based RL training.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Ludic Framework                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Agent r1   │  │   Agent r2   │  │   Agent rN   │  │
│  │  (LLM+Parser)│  │  (LLM+Parser)│  │  (LLM+Parser)│  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│         └─────────────────┼─────────────────┘           │
│                           │                             │
│                ┌──────────▼──────────┐                  │
│                │ MultiAgentProtocol  │                  │
│                └──────────┬──────────┘                  │
└───────────────────────────┼──────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │   SereLudicEnv        │
                │   (Wrapper)           │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │     PDDLEnv           │
                │  (Core Environment)   │
                └───────────────────────┘
```

### Key Components

#### 1. SereLudicEnv (`integrations/ludic/ludic_env.py`)

The Ludic wrapper that:
- Maps each robot to a Ludic agent ID (sorted alphabetically)
- Enforces joint action requirements in multi-agent mode
- Translates SERE observations to Ludic format
- Handles episode lifecycle (reset, step, done)

```python
from integrations.ludic import SereLudicEnv, make_ludic_env

# Create environment
env, meta = make_ludic_env(
    domain_path=None,
    task_path="kitchen/t11_multi_agent_parallel_brew.yaml",
    agent_ids=None,  # Auto-infer from robots
    force_interactive=True,
)

# Agent IDs correspond to robot symbols
print(env.agent_ids)  # ['r1', 'r2']
```

#### 2. Action Parsers (`integrations/ludic/ludic_parser.py`)

Parsers convert LLM outputs to SERE actions:

```python
from integrations.ludic import pddl_action_parser, pddl_action_tag_parser

# Raw PDDL parser
parser = pddl_action_parser()
result = parser("(move r1 kitchen pantry)")
# result.action = "(move r1 kitchen pantry)"

# XML-tag wrapped parser
tag_parser = pddl_action_tag_parser(tag="action")
result = tag_parser("<action>(move r1 kitchen pantry)</action>")
# result.action = "(move r1 kitchen pantry)"
```

#### 3. Ludic Agent

A Ludic `Agent` encapsulates:
- **Inference client**: Connection to LLM (e.g., vLLM)
- **Context strategy**: Memory management (full dialog, truncated thinking, etc.)
- **Parser**: Action extraction from LLM output
- **State**: Conversation history and internal state

```python
from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient

agent = Agent(
    client=VLLMChatClient(host="127.0.0.1", port=8000),
    model="Qwen/Qwen2.5-7B-Instruct",
    ctx=FullDialog(),
    parser=pddl_action_parser(),
)
```

### Multi-Agent Protocol

For multi-agent scenarios, use `MultiAgentProtocol`:

```python
from ludic.interaction import MultiAgentProtocol

agents = {
    "r1": Agent(client=client, model=model, ctx=ctx, parser=parser),
    "r2": Agent(client=client, model=model, ctx=ctx, parser=parser),
}

protocol = MultiAgentProtocol(agents)
```

---

## Creating and Using Agents

### Step 1: Define Domain with Robot Type

In your domain YAML (`src/sere/assets/domain/your_domain.yaml`):

```yaml
domain: your_domain
types:
  - name: robot
  - name: location
  - name: object
  # ... other types

actions:
  - name: move
    params:
      - {name: r, type: robot}
      - {name: from, type: location}
      - {name: to, type: location}
    pre:
      - "(at ?r ?from)"
      - "(adjacent ?from ?to)"
    add:
      - "(at ?r ?to)"
    del:
      - "(at ?r ?from)"
  # ... other actions
```

### Step 2: Define Task with Agents

In your task YAML (`src/sere/assets/tasks/your_domain/your_task.yaml`):

```yaml
id: your_task
name: Your Task Name

meta:
  domain: your_domain
  multi_agent: true  # Enable multi-agent mode
  max_steps: 20

objects:
  r1: robot
  r2: robot
  kitchen: location
  pantry: location
  # ... other objects

init:
  - (at r1 kitchen)
  - (at r2 pantry)
  # ... initial facts

termination:
  - name: goal
    all:
      - "(goal-condition-1)"
      - "(goal-condition-2)"
    outcome: success
    reward: 1.0
```

### Step 3: Load and Run

#### Direct PDDLEnv Usage

```python
from sere.io.task_loader import load_task
from sere.core.pddl_env.run_mode import RunMode

# Load environment
env, meta = load_task(
    domain_path=None,
    task_path="your_domain/your_task.yaml",
    run_mode=RunMode.INTERACTIVE,
)

# Reset
obs, info = env.reset()
print(obs)

# Step (multi-agent)
if env.multi_agent:
    actions = "(move r1 kitchen pantry)(idle r2)"
else:
    actions = "(move r1 kitchen pantry)"

obs, reward, done, info = env.step(actions)
```

#### Ludic Integration

```python
from integrations.ludic import make_ludic_env, pddl_action_parser
from ludic.agent import Agent
from ludic.interaction import MultiAgentProtocol

# Load environment
env, meta = make_ludic_env(
    domain_path=None,
    task_path="your_domain/your_task.yaml",
)

# Create agents
parser = pddl_action_parser()
agents = {
    agent_id: Agent(
        client=client,
        model=model,
        ctx=make_context(),
        parser=parser,
    )
    for agent_id in env.agent_ids
}

protocol = MultiAgentProtocol(agents)

# Run episode
outcomes = await protocol.run(
    env=env,
    max_steps=20,
)
```

---

## Agent Action Format

### Single Action

```
(action-name arg1 arg2 ...)
```

### Multiple Sequential Actions (BATCH mode)

```
(action1 arg1)(action2 arg2)(action3 arg3)
```

### Multi-Agent Joint Action (INTERACTIVE mode)

One action per robot, in any order:
```
(action1 r1 ...)(action2 r2 ...)
```

Or use `idle` for no-op:
```
(move r1 kitchen pantry)(idle r2)
```

### Special Actions

#### Idle (No-op)

```
(idle r1)
```

Does nothing but consumes a time step. Useful in multi-agent scenarios when one robot needs to wait.

---

## Examples

### Example 1: Single-Agent Task

**Task**: Robot must brew tea (see `t01_one_step_steep.yaml`)

```python
from sere.io.task_loader import load_task

env, _ = load_task(None, "kitchen/t01_one_step_steep.yaml")
obs, info = env.reset()

# Execute single action
obs, reward, done, info = env.step("(steep-tea r1 leaf1 mug1)")

if done and info["outcome"] == "success":
    print("Task completed!")
```

### Example 2: Multi-Agent Parallel Execution

**Task**: Two robots collaborate to brew tea (see `t11_multi_agent_parallel_brew.yaml`)

```python
from sere.io.task_loader import load_task

env, _ = load_task(None, "kitchen/t11_multi_agent_parallel_brew.yaml")
obs, info = env.reset()

# Robots work in parallel
steps = [
    "(open r1 kettle)(pick-up r2 leaf)",
    "(fill-water r1 kettle sink)(move r2 pantry kitchen)",
    "(close r1 kettle)(open r2 cup)",
    "(power-on r1 kettle)(put-in r2 leaf cup)",
    "(heat-kettle r1 kettle 4)(idle r2)",
    "(pour r1 kettle cup)(idle r2)",
    "(steep-tea r1 leaf cup)(idle r2)",
    "(move r1 kitchen table)(move r2 kitchen table)",
]

for actions in steps:
    obs, reward, done, info = env.step(actions)
    if done:
        print(f"Episode finished: {info['outcome']}")
        break
```

### Example 3: LLM-based Multi-Agent with Ludic

See `scripts/ludic/eval.py` for a complete example:

```python
from integrations.ludic import make_ludic_env, pddl_action_parser
from ludic.agent import Agent
from ludic.interaction import MultiAgentProtocol
from ludic.eval.core import run_eval_sync

# Setup
env, meta = make_ludic_env(
    domain_path=None,
    task_path="kitchen/t11_multi_agent_parallel_brew.yaml",
)

parser = pddl_action_parser()
agents = {
    agent_id: Agent(
        client=client,
        model="Qwen/Qwen2.5-7B-Instruct",
        ctx=context_strategy,
        parser=parser,
    )
    for agent_id in env.agent_ids
}

protocol = MultiAgentProtocol(agents)

# Evaluate
records, metrics = run_eval_sync(
    engine=rollout_engine,
    requests=rollout_requests,
    max_steps=40,
    concurrency=16,
)

print(f"Success rate: {metrics['success_rate']:.2%}")
```

### Example 4: Assembly Domain (Multi-Agent Fastening)

**Task**: Two robots fasten parts in parallel (see `t06_multi_agent_parallel_fasten.yaml`)

```python
from sere.io.task_loader import load_task

env, _ = load_task(None, "assembly/t06_multi_agent_parallel_fasten.yaml")
obs, info = env.reset()

# Both robots pick up tools
obs, reward, done, info = env.step(
    "(pick-up r1 wrench1)(pick-up r2 wrench2)"
)

# Both robots fasten parts simultaneously
obs, reward, done, info = env.step(
    "(fasten r1 part1 base1 wrench1)(fasten r2 part2 base2 wrench2)"
)
```

---

## Best Practices

### 1. Agent Design

- **Single-agent**: Use for tasks requiring sequential reasoning or when parallelism isn't beneficial
- **Multi-agent**: Use when tasks can be decomposed into parallel sub-tasks
- **Idle actions**: Always provide `(idle r)` for agents not acting in a turn

### 2. Action Coordination

In multi-agent mode:
- **Avoid conflicts**: Don't have two robots manipulate the same object simultaneously
- **Leverage parallelism**: Split independent tasks across robots
- **Use reference plans**: Validate multi-agent behavior with `reference_plan` in task YAML

Example reference plan:
```yaml
reference_plan:
  - (action1 r1 ...)(action2 r2 ...)  # Step 1: both act
  - (action3 r1 ...)(idle r2)         # Step 2: only r1 acts
```

### 3. Energy Management

If using numeric fluents with energy:
- Monitor `energy(r)` values in observations
- Use `(recharge r)` action at locations with `(has-charger l)`
- Avoid energy depletion (leads to episode termination)

### 4. Debugging

Enable debug mode:
```python
env.debug = True  # Prints detailed execution traces
```

Check `info` dict after each step:
```python
obs, reward, done, info = env.step(actions)
print(info["outcome"])        # success / invalid_move / timeout
print(info["messages"])       # Feedback messages
print(info.get("error"))      # Error details
```

### 5. LLM Training

When using Ludic for RL training:
- **Context strategy**: Use `TruncatedThinkingContext` for long episodes
- **Parsers**: Use `pddl_action_tag_parser` for better format control
- **Reward shaping**: Define `reward_shaping` in task YAML for milestone rewards
- **Evaluation**: Run periodic evals with `run_eval_sync` to track progress

Example reward shaping:
```yaml
reward_shaping:
  mode: instant  # or potential
  milestones:
    - expr: "(holding r1 mug1)"
      reward: 0.1
      once: true
    - expr: "(tea-ready mug1)"
      reward: 0.5
      once: true
```

### 6. Termination Conditions

Define clear success conditions:
```yaml
termination:
  - name: goal
    all:  # All conditions must be satisfied
      - "(predicate1)"
      - "(predicate2)"
    outcome: success
    reward: 1.0

  - name: failure
    when: "(bad-state)"
    outcome: failed
    reward: -1.0
```

Use `all`, `any`, or `when` for complex logic:
- `all`: Conjunction (AND)
- `any`: Disjunction (OR)
- `when`: Single condition

---

## Advanced Topics

### Custom Invariants

Define domain-specific constraints using `InvariantPlugin`:

```python
from sere.core.invariants import InvariantPlugin

class MyInvariant(InvariantPlugin):
    def validate(self, world: WorldState, static_facts: set) -> List[str]:
        errors = []
        # Check custom constraints
        if some_condition_violated:
            errors.append("Custom constraint violated")
        return errors

env, _ = load_task(..., plugins=[MyInvariant()])
```

### Visibility Scopes

Control what agents observe:

```python
formatter_config = {
    "visibility": "room",  # Only see facts in same room
    "observer_robot": "r1",  # Observer perspective
}

env, _ = load_task(..., formatter_config=formatter_config)
```

### Stochastic Actions

Enable stochastic outcomes:

```python
env, _ = load_task(..., enable_stochastic=True)
```

In domain YAML:
```yaml
actions:
  - name: grasp
    params: [{name: r, type: robot}, {name: o, type: object}]
    outcomes:
      - prob: 0.8
        add: ["(holding ?r ?o)"]
      - prob: 0.2
        add: ["(dropped ?o)"]
```

### Derived Predicates

Define higher-level predicates computed from base facts:

```yaml
derived:
  - name: tea-ready
    args: [{name: c, type: container}]
    when:
      - "(in ?l ?c)"
      - "(steeped ?l)"
    nl:
      - "tea is ready in {c}"
```

---

## Summary

SERE's agent architecture supports:
- ✅ Single-agent and multi-agent modes
- ✅ PDDL-style grounded actions
- ✅ Parallel execution with joint actions
- ✅ Seamless Ludic integration for LLM-RL
- ✅ Rich observations with natural language
- ✅ Flexible termination and reward shaping
- ✅ Energy management and numeric fluents
- ✅ Stochastic outcomes and derived predicates

For more examples, see:
- `src/sere/assets/tasks/kitchen/` – Kitchen domain tasks
- `src/sere/assets/tasks/assembly/` – Assembly domain tasks
- `scripts/ludic/eval.py` – LLM-based multi-agent evaluation
- `tests/test_ludic_integration.py` – Integration tests

Happy agent building! 🤖
