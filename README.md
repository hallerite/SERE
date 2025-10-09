# Symbolic Embodied Reasoning Environments (SERE)

**SERE** is a lightweight framework for building **symbolic, embodied reasoning environments** — where agents must manipulate objects, respect spatial and causal constraints, and satisfy task goals expressed in PDDL-style logic.

It’s designed for **RL + LLM training**, giving you a **Gym-style API** but with symbolic state, grounded actions, stochasticity, and reward shaping built in.

---

## ✨ Features

- **YAML-defined domains** – types, predicates, fluents, actions (with preconditions, add/del, conditional and numeric effects).  
- **PDDL-style grounding** – parses `(pick-up r1 mug1)` into concrete state updates.  
- **World state engine** – maintains objects, facts, numeric fluents, and enforces invariants.  
- **Numeric fluents, durations, energy** – model time, resources, and stochastic outcomes.  
- **Reward shaping & termination rules** – instant milestones, potential-based shaping, and flexible episode outcomes.  
- **Invariant plugins** – register domain-specific constraints (e.g. “object can’t be in two places”).  
- **Human-readable rendering** – natural language + PDDL observations for LLM prompting, with affordance lists.  
- **Reference plans & regression tests** – validate domains and ensure backward compatibility.  

---

## 🏗 Architecture

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
├── pddl/                    # Domain parsing, grounding, NL mapping
├── io/                      # Task loader utilities
├── cli/                     # Command-line runner
│   └── run_task.py
└── assets/
    ├── domain/              # Domain YAMLs (kitchen, assembly, …)
    └── tasks/               # Task YAMLs (per domain)
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
python -m sere.cli.run_task kitchen/t01_one_step_steep.yaml
```

Note: `cli.run_task` looks at `src/sere/assets/tasks/` for the `yaml` file.

You’ll see output like:

```
...
State:
  (at r1 hallway)
  (obj-at kettle1 kitchen)
  (clear-hand r1)
Goal:
  (tea-ready mug1)

Reply with (action args).
```

Example step:

```
(move r1 hallway kitchen)
```

The environment will parse and apply the action, update time/energy, and return the next observation plus reward.

---

## 🛠 Authoring Domains & Tasks

- **Domains** (`assets/domain/*.yaml`) define:
  - **Types** (`robot`, `location`, `object`, …)  
  - **Predicates** (`at`, `holding`, `in`, …)  
  - **Fluents** (`energy`, `time`, …)  
  - **Actions** (with preconditions, add/del, conditional effects, stochastic outcomes, numeric updates, durations)

- **Tasks** (`assets/tasks/**/*.yaml`) define:
  - **Objects** with types  
  - **Initial state** (facts + fluent values)  
  - **Statics** (e.g. adjacency graph)  
  - **Goals** (logical literals or conditions)  
  - **Optional shaping rules** and **reference plans**  

This separation makes it easy to randomize tasks or auto-generate curricula.

## To Do
- improve docs
- add domain randomization for tasks
    - multiple different distinct outcomes for some actions