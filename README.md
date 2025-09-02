# Symbolic Embodied Reasoning Environments (SERE)

## Architecture
SERE/
├─ domain/
│  ├─ kitchen_domain.yaml
│  └─ ... # (add more later)
├─ tasks/
│  └─ kitchen/
│     └─ t01_make_tea_basic.yaml
├─ src/
│  ├─ pddl/
│  │  ├─ domain_spec.py                # YAML → DomainSpec (predicates/actions/types/NL)
│  │  ├─ grounding.py                  # parse '(act ...)', instantiate pre/add/del
│  │  └─ nl_mapper.py                  # literals/actions → natural language strings
│  ├─ core/
│  │  ├─ world_state.py                # typed constants, fact set, invariants, PDDL render
│  │  ├─ invariants.py                 # plugin interface + generic checks
│  │  └─ pddl_env.py                   # RL-style env: reset/step/terminal/rewards
│  ├─ io/
│  │  ├─ task_loader.py                # load a task YAML → WorldState + Env
│  │  └─ factory.py                    # convenience loader w/ domain + plugins
│  └─ cli/
│     └─ run_task.py                   # quick demo runner
└─ tests/
   └─ test_reference_plan.py           # sanity test: solve t01 with a fixed plan
