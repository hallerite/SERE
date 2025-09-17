# Symbolic Embodied Reasoning Environments (SERE)

**SERE** is a lightweight framework for building **symbolic, embodied reasoning environments** — where agents must manipulate objects, respect spatial constraints, and satisfy task goals expressed in PDDL-style logic.

**SERE** exposes a Gym-like API to do RL training with LLMs!

## Features

- **YAML-defined domains**: types, predicates, fluents, actions (with pre/add/del/num_eff).  
- **PDDL grounding engine**: parses actions like `(pick-up r1 mug1)` into concrete effects.  
- **World state tracker**: maintains typed objects, facts, invariants, and numeric fluents.  
- **RL-style environment**: `reset` / `step` API with rewards and terminal outcomes.  
- **Plugin system**: inject domain-specific invariants (e.g. object can’t be in two places).  
- **Tests baked in**: invariants, edge-cases, and reference plans.  

---

## Architecture

```
SERE/
├── domain/                 # Domain specifications (YAML)
│   ├── kitchen_domain.yaml
│   └── ...
│
├── tasks/                  # Task instances (initial state + goals)
│   └── kitchen/
│       └── t01_make_tea_basic.yaml
│
├── src/
│   ├── pddl/
│   │   ├── domain_spec.py      # YAML → DomainSpec (preds/actions/types/NL)
│   │   ├── grounding.py        # Parse '(act ...)', instantiate pre/add/del
│   │   └── nl_mapper.py        # Map literals/actions → natural language
│   │
│   ├── core/
│   │   ├── world_state.py      # Typed constants, fact set, invariants, PDDL render
│   │   ├── invariants.py       # Plugin interface + generic checks
│   │   └── pddl_env.py         # RL-style Env: reset/step/terminal/rewards
│   │
│   ├── io/
│   │   ├── task_loader.py      # Load task YAML → WorldState + Env
│   │   └── factory.py          # Convenience loaders w/ domain + plugins
│   │
│   └── cli/
│       └── run_task.py         # Demo runner (step through a task)
│
└── tests/
    ├── kitchen/
    │   └── test_core_kitchen.py   # Unit tests for kitchen semantics
    └── test_reference_plan.py     # Sanity check: solve t01 with a fixed plan
```

---

## Installation

```bash
git clone https://github.com/yourname/SERE.git
cd SERE
uv venv .venv
source .venv/bin/activate
uv sync
```

Requires **Python 3.11+**.  

---

## Running a Task

From the repo root:

```bash
python -m src.cli.run_task tasks/kitchen/t01_make_tea_basic.yaml
```

You’ll see output like:

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

## Writing Domains

Domains are YAML files that define:

- **Types** (`robot`, `location`, `object`, …)  
- **Predicates** (`at`, `holding`, `in`, …)  
- **Fluents** (`energy`, `time` …)  
- **Actions** with preconditions/effects  

See [`domain/kitchen_domain.yaml`](domain/kitchen_domain.yaml) for a full example.

---

## Writing Tasks

Tasks specify:

- **Objects** with types  
- **Initial state** (facts + fluent values)  
- **Goals**  
- **Statics** (like adjacency graph)  

See [`tasks/kitchen/t01_make_tea_basic.yaml`](tasks/kitchen/t01_make_tea_basic.yaml).  

---

## Roadmap

- More domains (warehouse, assembly)
- Natural-language task descriptions in the env obs