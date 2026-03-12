# PDDL Benchmarks & Planning Environments for LLM RL Training
## Comprehensive Landscape Research

---

## 1. Major PDDL Benchmark Collections

### 1.1 International Planning Competition (IPC) Archives

The IPC has been running since 1998 and is the primary source of standardized PDDL benchmarks. Each competition introduces new domains and problem instances.

| IPC Year | Domains | Variants | Track |
|----------|---------|----------|-------|
| 1998     | 7       | 14       | Classical |
| 2000     | 5       | 12       | Classical |
| 2002     | 8       | 48       | Classical |
| 2004     | 8       | 47       | Deterministic |
| 2006     | 7       | 50       | Deterministic |
| 2008     | 11      | 41       | Deterministic |
| 2011     | 19      | 54       | Deterministic |
| 2014     | 23      | 66       | Deterministic |
| 2018     | 22      | 33       | Deterministic |
| 2020     | 23      | 33       | Deterministic |
| 2023     | 17      | 20       | Deterministic |

**Key Repositories:**
- **potassco/pddl-instances** (GitHub): Covers IPC 1998-2014 with consistent structure. Each domain has `domain.pddl` and `instances/instance-x.pddl`.
- **plaans/tyr-ipc-domains** (GitHub): Extended coverage through IPC 2023.
- **ipc2023-classical/ipc2023-dataset** (GitHub): IPC 2023 specific dataset with 7 new domains.

### 1.2 IPC 2023 Classical Track Domains (Most Recent)

Seven newly designed domains:
1. **Folding** - Protein/paper folding
2. **Labyrinth** - Maze navigation
3. **Quantum Circuit Layout Synthesis** - (Won Outstanding Domain Submission Award)
4. **Recharging Robots** - Robot coordination with energy constraints
5. **Ricochet Robots** - Sliding puzzle variant
6. **Rubik's Cube** - 3D combinatorial puzzle
7. **Slitherlink** - Logic puzzle

PDDL features used: STRIPS, action costs, negative preconditions, conditional effects, forall quantifiers, disjunctions, negative goal conditions. Six of seven come with task generators.

### 1.3 IPC 2023 Learning Track Domains

Ten domains used for training/evaluating learning-based planners:
1. **Blocksworld** - Block stacking (classic, well-studied)
2. **Childsnack** - Preparing/serving food
3. **Ferry** - Transporting cars across a river
4. **Floortile** - Painting floor tiles
5. **Miconic** - Elevator scheduling
6. **Rovers** - Mars rover planning
7. **Satellite** - Satellite observation scheduling
8. **Sokoban** - Box-pushing puzzle (PSPACE-complete)
9. **Spanner** - Tool retrieval (polynomial)
10. **Transport** - Vehicle routing

Each domain provides 50-100 training tasks in ascending difficulty, plus generators for creating more.

### 1.4 planning.domains API

The planning.domains initiative provides three components:
- **api.planning.domains**: Programmatic REST API access to all existing PDDL planning problems. Collections for every IPC plus individual planner collections.
- **editor.planning.domains**: Fully featured online PDDL editor with syntax highlighting, auto-completion, and solver integration.
- **solver.planning.domains**: Cloud-based solver hosting multiple planners from the planutils project.

---

## 2. Well-Studied PDDL Domains (Taxonomy)

### 2.1 Easy/Introductory Domains (often polynomial or short plans)

| Domain | Description | Actions | Key Characteristics |
|--------|-------------|---------|-------------------|
| **Gripper** | Robot with grippers transporting balls between rooms | 3 (pick, move, drop) | 7 predicates, very simple structure |
| **Ferry** | Transport cars across river on a ferry | 3-4 (board, sail, debark) | Linear structure, small branching factor |
| **Miconic** | Elevator picking up/dropping off passengers | 4 (up, down, board, depart) | Simple scheduling |
| **Spanner** | Collecting tools and tightening nuts | ~3 | Polynomial optimal planning |
| **Movie** | Watching a movie (getting snacks, etc.) | Few | Very simple conjunction of subgoals |

### 2.2 Medium Domains (NP-hard, moderate plan lengths)

| Domain | Description | Actions | Key Characteristics |
|--------|-------------|---------|-------------------|
| **Blocksworld** | Stacking/unstacking blocks on a table | 4 (pick-up, put-down, stack, unstack) | The canonical planning benchmark. ~5 predicates. |
| **Logistics** | Trucks and planes moving packages between locations/cities | 6 (load/unload truck/airplane, drive, fly) | Multi-level transportation |
| **Depot** | Combining transportation + blocksworld (crate stacking on pallets) | ~5 | Hybrid domain |
| **Satellite** | Scheduling observations with satellite instruments | ~5 | Resource management |
| **Rovers** | Mars rover navigation, sampling, communication | ~8 | Multi-agent feel, resource constraints |
| **Childsnack** | Making and serving sandwiches | ~4 | Resource tracking |
| **Transport** | Vehicle routing with fuel/capacity | ~4 | Constrained optimization |

### 2.3 Hard Domains (PSPACE-complete, long plans, irreversible actions)

| Domain | Description | Actions | Key Characteristics |
|--------|-------------|---------|-------------------|
| **Sokoban** | Push boxes onto target locations | 4 (move in 4 directions) | PSPACE-complete, irreversible moves, deadlock states |
| **Rubik's Cube** | Solving Rubik's cube | 12-18 (face rotations) | Extremely large state space |
| **Freecell** | Card game solitaire | ~10 | Large branching factor, deep plans |
| **Floortile** | Painting tiles while navigating | ~6 | Coordination constraints |
| **N-Puzzle** | Sliding tile puzzle | 4 (slide directions) | NP-hard for optimal, large state space |
| **Visitall** | Visit all grid cells | 4+ | Hamiltonian path-like |

### 2.4 Computational Complexity Summary

- **General PDDL/STRIPS planning**: Plan existence is **PSPACE-complete**
- **Bounded plan length**: PSPACE-hard; NP-hard if negative effects are disallowed
- **No negative preconditions**: PlanSAT is in P, but optimal planning still NP-hard
- **Domain-specific**: Ranges from polynomial (Spanner, some Ferry variants) to PSPACE-complete (Sokoban)
- **Temporal PDDL extensions**: PSPACE-complete (no self-overlap) to EXPSPACE-complete (with overlap) to undecidable (non-zero separation)

---

## 3. PDDL Problem Generators and Datasets

### 3.1 AI-Planning/pddl-generators (GitHub)

**70+ domain generators** including:
- agricola, assembly, barman, blocksworld, briefcaseworld, cavediving, childsnack, citycar, crewplanning, data-network, delivery, depots, driverlog, elevators, ferry, floortile, freecell, fridge, goldminer, grid, gripper, grippers, hanoi, hiking, logistics, maintenance, matchcellar, miconic (3 variants), minigrid, movie, mprime, mystery, nomystery, npuzzle, nurikabe, openstacks, parking, pathways, pegsol, rovers, satellite, scanalyzer, schedule, snake, sokoban, spanner, spider, storage, termes, tetris, tidybot, tpp, transport, trucks, tsp, turnandopen, tyreworld, visitall, woodworking, zenotravel

Each generator has domain-specific parameters to control:
- Number of objects (blocks, packages, rooms, etc.)
- Problem size / grid dimensions
- Number of goals
- Resource constraints

### 3.2 PDDLFuse (2024)

A tool for generating **novel, diverse** planning domains (not just problem instances within existing domains). Key innovation: domain randomization for planning, inspired by sim-to-real RL. Can modulate difficulty of generated domains. Useful for training generalizable planners.

### 3.3 Key Datasets for LLM Research

| Dataset | Size | Purpose | Domains |
|---------|------|---------|---------|
| **Planetarium** | 145,918 text-PDDL pairs | NL-to-PDDL translation | 73 unique state combinations, 4 base domains |
| **PlanBench** | Extensible | Plan generation/verification by LLMs | Blocksworld + extensible |
| **ACPBench** | Large (generated) | 7 reasoning tasks for LLMs | 13 domains incl. Blocksworld, Logistics, Rovers, etc. |
| **AutoPlanBench** | 12 custom + 33 IPC domains | NL planning from PDDL auto-conversion | 45 total datasets |
| **Proc2PDDL** | 27 domain + 81 problem files | Open-domain procedural text to PDDL | 27 WikiHow-derived domains |

---

## 4. PDDL Environments for RL/ML

### 4.1 PDDLGym / pddlgymnasium

**The primary PDDL-to-RL bridge.** Converts any PDDL domain into an OpenAI Gym/Gymnasium environment.

- **Repository**: github.com/tomsilver/pddlgym (original), pddlgymnasium on PyPI (updated Gymnasium API)
- **PDDL support**: PDDL 1.2 subset (STRIPS, typing, quantifiers, disjunctions, equality, derived predicates). No conditional effects or action costs.
- **Included domains**: 14+ (Sokoban, Depot, Blocks, Hanoi, Snake, Gripper, Ferry, Elevator, TSP, Minecraft, etc.) + probabilistic variants (River, Triangle Tireworld, Exploding Blocks)
- **Observation space**: Relational tuples (literals frozenset, objects frozenset, goal Literal)
- **Action space**: Configurable - operators_as_actions=True uses PDDL operators directly; dynamic_action_space=True filters to valid actions only
- **Adding custom domains**: Place domain.pddl in `pddl/` directory, problem files in subdirectory, register in `__init__.py`

### 4.2 pyRDDLGym

For **probabilistic/MDP planning** (not PDDL but closely related RDDL language):
- Generates Gym environments from RDDL (Relational Dynamic Influence Diagram Language) descriptions
- Used in IPC 2023 Probabilistic & RL Track
- Supports MDPs, POMDPs, factored MDPs
- Standard Gym interface for RL agents

### 4.3 GOOSE (Graph Learning for Planning)

Neural network-based heuristic learning system:
- Represents PDDL problems as graphs (nodes = objects/facts, edges = relations)
- Uses Weisfeiler-Leman algorithm for graph embeddings
- Trains on 50,000 examples across 50 planning domains
- WL-GOOSE outperforms hFF heuristic with 2-3 orders of magnitude fewer parameters

### 4.4 Using PDDL Directly as RL Environment for LLMs

The key insight from recent research: PDDL can serve as a **text-based RL environment** for LLMs where:
- **State**: PDDL state description (set of true predicates)
- **Action space**: Grounded PDDL actions applicable in current state
- **Transition**: PDDL action effects applied to state
- **Reward**: Binary (goal achieved or not) or shaped (partial goal satisfaction, plan length penalty)
- **Verification**: VAL validator provides ground-truth correctness checking

---

## 5. Tools for PDDL Validation and Plan Verification

### 5.1 VAL (Plan Validator)

- **Repository**: github.com/KCL-Planning/VAL
- **Purpose**: Validates plans against domain/problem files
- **Features**: Plan validation, PDDL parsing/error reporting, grounding
- **Usage**: `Validate -t <timeout> -v <domain> <problem> <plan>`
- **Critical for RL**: Can be used as a **verifier reward signal** in training loops (see Section 6)

### 5.2 Fast Downward

- **Purpose**: State-of-the-art classical planner using heuristic search
- **PDDL support**: Propositional fragment of PDDL 2.2 including ADL, derived predicates
- **Architecture**: Translates PDDL to SAS+ (multivalued planning tasks), then searches
- **Heuristics**: FF, CEA, pattern databases, Cartesian abstractions, landmark heuristics
- **Key features**: Preferred operators, deferred evaluation, multi-heuristic search
- **Configurations**: LAMA (satisficing), Scorpion (optimal), many more

### 5.3 Pyperplan

- **Repository**: github.com/aibasel/pyperplan
- **Purpose**: Lightweight STRIPS planner in Python (educational/prototyping)
- **Features**: Clean Python code, multiple search algorithms (BFS, A*, greedy best-first)
- **Good for**: Integration with Python ML pipelines, easy to modify

### 5.4 Unified Planning Framework

- **Repository**: github.com/aiplan4eu/unified-planning
- **Purpose**: Python library providing high-level API for planning
- **Features**: PDDL 2.1 level 3 parser, multiple planner backends, problem transformation
- **Planner integration**: PDDLPlanner, PDDLAnytimePlanner base classes
- **PDDL I/O**: Parse domain/problem files, write PDDL, read plans

### 5.5 planutils

- **Repository**: github.com/AI-Planning/planutils
- **Docker image**: aiplanning/planutils:latest
- **Purpose**: Collection of planners installable via CLI
- **Includes**: VAL, Fast Downward, LAMA, POPF, planning.domains integration
- **Usage**: `planutils install lama`, `planutils run lama domain.pddl problem.pddl`

### 5.6 Other PDDL Libraries

| Library | Language | Purpose |
|---------|----------|---------|
| **pddl** (PyPI) | Python | Complete PDDL 3.1 parser |
| **PDDL.jl** | Julia | Parser, interpreter, compiler |
| **INVAL** | Python | Plan validator + PDDL tools |
| **pddl-lib** | Python | Simple PDDL parsing interface |

---

## 6. Existing Work on LLMs for PDDL Planning

### 6.1 Direct Plan Generation

| System | Approach | Key Finding |
|--------|----------|-------------|
| **PlanBench** (Valmeekam et al., 2023) | Evaluate LLMs on plan generation/verification | LLMs perform poorly on plan generation even with CoT |
| **AutoPlanBench** (Stein & Koller, 2024) | Auto-convert PDDL to NL, test LLM planning | LLMs match manual NL conversion quality; broad evaluation across 33+ IPC domains |
| **o1 on PlanBench** (Valmeekam et al., 2024) | Evaluate OpenAI o1 (LRM) | Quantum improvement over LLMs but far from saturating benchmark; 27% false positive on impossibility |

### 6.2 LLM as PDDL Formalizer (NL to PDDL)

| System | Approach | Key Finding |
|--------|----------|-------------|
| **LLM+P** (Liu et al., 2023) | LLM maps NL to PDDL problem file, classical planner solves | Best of both worlds: LLM understanding + planner guarantees |
| **NL2Plan** (2024) | Fully automatic NL to complete PDDL (domain + problem) | First system to auto-generate full PDDL from minimal text |
| **Planetarium** (2024) | Benchmark for NL-to-PDDL | GPT-4o: 96% parseable, 94% solvable, only 25% semantically correct |

### 6.3 LLM-Enhanced Planning

| System | Approach | Key Finding |
|--------|----------|-------------|
| **SayCanPay** (Hazra et al., 2024) | LLM generates actions (Say), models check feasibility (Can) and reward (Pay) | Domain knowledge improves over SayCan |
| **PDDL-INSTRUCT** (2025) | Two-stage instruction tuning: CoT + iterative validator feedback | Teaches LLMs precondition-effect reasoning chains |
| **LLM-Generated Heuristics** (Correa et al., 2025) | LLMs generate Python heuristic functions from PDDL, used in GBFS | Competitive with LAMA; outperforms hFF; beats best learning-based planner |
| **End-to-End Agentic Framework** (2025) | LangGraph orchestrator + 8 specialized agents for PDDL generation/refinement | 30-40% improvement on Blocksworld, logistics, Hanoi over baselines |

### 6.4 RL Training with PDDL (Most Relevant to Your Use Case)

**"On the Generalization Gap in LLM Planning" (2025)** - Key paper for PDDL+LLM+RL:
- **Setup**: Fine-tuned 1.7B LLM on 40,000 domain-problem-plan tuples from 10 IPC domains
- **Training domains**: Ferry, Floortile, Blocksworld, Childsnack, Spanner, Satellite, Maintenance, Parking, Transport, Miconic
- **RL method**: Group Relative Policy Optimization (GRPO) with VAL verifier reward
- **Reward structure**:
  - +1.0 for valid plans achieving goals
  - +0.1 for syntactically valid plans not reaching goals
  - -0.1 for execution failures (unsatisfied preconditions)
  - 0.0 for syntax errors
- **Results**: 82.9% in-domain validity, **0% cross-domain** (Rover, Briefcase)
- **Key finding**: Models learn domain-specific patterns, not transferable planning competence. Symbol anonymization causes >10% drops.

**SafeGen-LLM (2025)**: PDDL with safety constraints
- Uses PDDL3 benchmarks with explicit safety constraints
- Two-stage training: SFT on constraint-compliant plans + GRPO with reward machines
- Focus on safety generalization

### 6.5 Frontier LLM Performance on PDDL (2025 Evaluation)

Evaluated on 8 IPC 2023 Learning Track domains (360 tasks):

| Planner | Standard Tasks Solved | Obfuscated Tasks |
|---------|----------------------|-----------------|
| **LAMA** (classical) | 204/360 | 204/360 (unchanged) |
| **GPT-5** | 205/360 | 152/360 (-26%) |
| **DeepSeek R1** | 157/360 | 93/360 (-41%) |
| **Gemini 2.5 Pro** | 155/360 | 146/360 (-6%) |

GPT-5 matches LAMA on standard tasks but degrades under obfuscation. Gemini 2.5 Pro is most robust to obfuscation. All LLMs are orders of magnitude less efficient than specialized planners.

---

## 7. PDDL Version Features Reference

| Version | IPC Year | Key Features Added |
|---------|----------|--------------------|
| **PDDL 1.0** | 1998 | STRIPS, typing, ADL, conditional effects, quantifiers |
| **PDDL 2.1** | 2002 | Numeric fluents, plan-metrics, durative/continuous actions |
| **PDDL 2.2** | 2004 | Derived predicates (axioms), timed initial literals |
| **PDDL 3.0** | 2006 | State-trajectory constraints, preferences |
| **PDDL 3.1** | 2008 | Object fluents, extended preferences |

For RL training, **PDDL 1.2 / STRIPS** is the sweet spot: well-supported by all tools, sufficient expressiveness for most classical domains, and fully supported by PDDLGym.

---

## 8. Recommended Architecture for PDDL-Based LLM RL Training

Based on this research, the recommended stack would be:

### Environment Layer
- **PDDLGym/pddlgymnasium** for Gym-compatible PDDL environments
- OR custom PDDL simulator using **pddl** (Python parser) + manual state tracking
- **AI-Planning/pddl-generators** for creating training instances at scale

### Verification Layer
- **VAL** for plan validation (can be used as reward signal)
- **Fast Downward** for generating reference solutions / oracle plans
- **Pyperplan** for lightweight Python-native plan checking

### Training Data
- **IPC 2023 Learning Track** (10 domains) as primary benchmark
- **pddl-generators** for generating 500+ instances per domain at varying difficulty
- **AutoPlanBench** for NL-PDDL paired data

### Difficulty Curriculum
Ordered roughly by difficulty:
1. **Tier 1 (Easy)**: Gripper, Ferry, Spanner, Miconic
2. **Tier 2 (Medium)**: Blocksworld, Childsnack, Transport, Satellite, Logistics
3. **Tier 3 (Hard)**: Rovers, Floortile, Depot, Parking
4. **Tier 4 (Very Hard)**: Sokoban, Rubik's Cube, Freecell, N-Puzzle

### Reward Design (from literature)
- **Binary**: +1 goal achieved, 0 otherwise
- **Graded (VAL-based)**: +1.0 valid goal-reaching plan, +0.1 valid non-goal plan, -0.1 precondition violation, 0.0 syntax error
- **Plan quality**: Reward inversely proportional to plan length vs. optimal
- **Partial credit**: Fraction of goal predicates achieved

---

## 9. Key Gaps and Opportunities

1. **Cross-domain generalization**: Current fine-tuned LLMs show 0% cross-domain transfer. This is the biggest open problem.
2. **Obfuscation robustness**: LLMs rely on semantic names rather than structural reasoning. Training with anonymized symbols could help.
3. **Scalability**: LLMs competitive on small instances but degrade on larger ones where classical planners excel.
4. **No existing PDDL RL environment specifically designed for LLM text-based interaction** - PDDLGym uses relational/symbolic observations, not text. A text-based wrapper would be needed.
5. **Limited PDDL coverage in PDDLGym**: No conditional effects or action costs support.
6. **Reward shaping**: Rich opportunity to use PDDL structure (landmarks, delete relaxation, goal distance) for reward shaping.

---

## Sources

### IPC and Benchmarks
- [potassco/pddl-instances](https://github.com/potassco/pddl-instances)
- [plaans/tyr-ipc-domains](https://github.com/plaans/tyr-ipc-domains)
- [AI-Planning/pddl-generators](https://github.com/AI-Planning/pddl-generators)
- [IPC 2023 Classical Tracks](https://ipc2023-classical.github.io/)
- [IPC 2023 Learning Tracks](https://ipc2023-learning.github.io/)
- [ipc2023-classical/ipc2023-dataset](https://github.com/ipc2023-classical/ipc2023-dataset)
- [IPC 2023 Overview Paper (AI Magazine)](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12169)

### PDDL Environments and Tools
- [PDDLGym](https://github.com/tomsilver/pddlgym) / [pddlgymnasium](https://pypi.org/project/pddlgymnasium/)
- [pyRDDLGym](https://arxiv.org/html/2211.05939v5)
- [VAL Plan Validator](https://github.com/KCL-Planning/VAL)
- [INVAL Validator](https://github.com/patrikhaslum/INVAL)
- [Fast Downward](https://www.fast-downward.org/)
- [Pyperplan](https://github.com/aibasel/pyperplan)
- [planutils](https://github.com/AI-Planning/planutils)
- [Unified Planning Framework](https://github.com/aiplan4eu/unified-planning)
- [planning.domains API](https://api.planning.domains/)
- [Planning.wiki](https://planning.wiki/)

### LLM + PDDL Benchmarks
- [PlanBench](https://github.com/karthikv792/LLMs-Planning)
- [ACPBench](https://arxiv.org/html/2410.05669)
- [Planetarium](https://github.com/BatsResearch/planetarium)
- [AutoPlanBench](https://github.com/minecraft-saar/autoplanbench)
- [Proc2PDDL](https://github.com/zharry29/proc2pddl)
- [PDDLFuse](https://arxiv.org/abs/2411.19886)

### Key Papers (2024-2025)
- [The 2025 Planning Performance of Frontier LLMs](https://arxiv.org/abs/2511.09378)
- [On the Generalization Gap in LLM Planning: Tests and Verifier-Reward RL](https://arxiv.org/abs/2601.14456)
- [LLMs Still Can't Plan; Can LRMs? (o1 evaluation)](https://arxiv.org/abs/2409.13373)
- [Classical Planning with LLM-Generated Heuristics](https://arxiv.org/abs/2503.18809)
- [PDDL-INSTRUCT: Enhancing Symbolic Planning](https://pulkitverma.net/assets/pdf/vlfms_lm4plan25/vlfms_lm4plan25.pdf)
- [An End-to-end Planning Framework with Agentic LLMs and PDDL](https://arxiv.org/abs/2512.09629)
- [A Modern Survey of LLM Planning Capabilities (ACL 2025)](https://aclanthology.org/2025.acl-long.958.pdf)
- [LLMs as Planning Formalizers Survey](https://arxiv.org/html/2503.18971v2)
- [GOOSE: Learning Domain-Independent Heuristics](https://github.com/DillonZChen/goose)
- [NL2Plan: Robust LLM-Driven Planning](https://arxiv.org/abs/2405.04215)
- [SayCanPay](https://arxiv.org/abs/2308.12682)
- [LLM+P](https://github.com/Cranial-XIX/llm-pddl)
- [SafeGen-LLM](https://arxiv.org/html/2602.24235)
- [Generalized Planning in PDDL Domains with Pretrained LLMs](https://arxiv.org/abs/2305.11014)
