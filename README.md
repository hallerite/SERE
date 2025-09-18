# Symbolic Embodied Reasoning Environments (SERE)

**SERE** is a lightweight framework for building **symbolic, embodied reasoning environments** — where agents must manipulate objects, respect spatial constraints, and satisfy task goals expressed in PDDL-style logic.

It’s designed for **RL + LLM training**, giving you a **Gym-style API** but with symbolic state, action grounding, reward shaping, and domain randomization baked in.

---

## ✨ Features

- **YAML-defined domains** – types, predicates, fluents, actions (with pre/add/del/num_eff/cond_eff).  
- **PDDL-style grounding** – parses `(pick-up r1 mug1)` into concrete state updates.  
- **World state engine** – maintains objects, facts, numeric fluents, and enforces invariants.  
- **Reward shaping + stochasticity** – attach per-action rewards, energy costs, or noisy outcomes.  
- **Curriculum & randomization** – parameterize tasks for scalable training.  
- **RL-style environment** – `reset` / `step` API with rewards, terminal checks, and episode info.  
- **Invariant plugins** – easy way to register domain-specific constraints (e.g. “object can’t be in two places”).  
- **Human-readable rendering** – map literals and actions back to natural language for LLM prompting.  
- **Reference plan testing** – validate tasks and regression-test domains automatically.  

---

## 🏗 Architecture

```
SERE/
├── src/
│   ├── sere/core/             # Core engine
│   │   ├── pddl_env.py        # RL-style Env: reset/step/reward/done
│   │   ├── world_state.py     # Facts, objects, numerics, goal checks
│   │   ├── actions.py         # Action application, conditional & stochastic effects
│   │   ├── invariants.py      # Generic + custom constraints
│   │   └── rewards.py         # Reward shaping utilities
│   │
│   ├── sere/pddl/             # Parsing + grounding
│   │   ├── domain_spec.py     # Load YAML domain spec → Domain object
│   │   ├── action_grounding.py# Generate all applicable actions for a state
│   │   └── nl_mapper.py       # Literal/action → natural language
│   │
│   ├── sere/io/
│   │   ├── task_loader.py     # Load task YAML → Env + initial state
│   │   └── factory.py         # Helpers for domain/task combos
│   │
│   └── sere/cli/
│       └── run_task.py        # Interactive REPL for debugging tasks
│
├── assets/
│   ├── domains/               # Domain definitions (kitchen, assembly, …)
│   └── tasks/                 # Task instances (YAML)
│
└── tests/                     # Reference plans + regression tests
```

---

## 🔧 Installation

```bash
git clone https://github.com/yourname/SERE.git
cd SERE
uv venv .venv
source .venv/bin/activate
uv sync
```

Requires **Python 3.11+**.

---

## ▶️ Running a Task

From the repo root:

```bash
python -m src.sere.cli.run_task assets/tasks/kitchen/t01_make_tea_basic.yaml
```

You’ll get output like:

```
State:
  (at r1 hallway)
  (obj-at kettle1 kitchen)
  (clear-hand r1)
Goal:
  (tea-ready mug1)
  (at r1 table)

Reply with <move>(action args)</move>.
```

Example step:

```xml
<move>(move r1 hallway kitchen)</move>
```

---

## 🛠 Authoring Domains & Tasks

- **Domains** (`assets/domains/*.yaml`) define:
  - **Types** (`robot`, `location`, `object`, …)
  - **Predicates** (`at`, `holding`, `in`, …)
  - **Fluents** (`energy`, `time`, …)
  - **Actions** (with preconditions, add/del effects, numeric updates, and conditional effects)

- **Tasks** (`assets/tasks/**/*.yaml`) define:
  - **Objects** with types
  - **Initial state** (facts + fluent values)
  - **Statics** (e.g. adjacency graph)
  - **Goals** (logical literals)
  - **Optional reward shaping** and **reference plans**

This separation makes it easy to swap domains or auto-generate curriculum tasks.