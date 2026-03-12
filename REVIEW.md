# SERE Codebase Review

**Original review:** 2026-01-13
**Last updated:** 2026-03-12

---

## Status

SERE has evolved significantly since the initial review. Key changes:

- **22 domains** (20 PDDL IPC benchmarks + 2 YAML), up from 2 YAML-only domains
- **300+ problem instances** across all domains, graded by difficulty
- **Agentic sandbox** -- miniSWE-style environment with tool use (bash, read/write files, validate, simulate)
- **Pure plan validator** -- standalone validation engine with step-by-step diagnostics
- **Verifiers integration** -- native `MultiTurnEnv` for `prime eval` (both agentic and pure PDDL modes)
- **NL mappings removed** -- raw PDDL used directly as prompts
- **Outcome-only reward** -- no step/invalid penalties by default

---

## Resolved Issues

| # | Issue | Resolution |
|---|-------|------------|
| 1 | No procedural task generation | Mitigated: 300+ IPC benchmark problems provide sufficient diversity |
| 2 | Only 2 domains | **Fixed**: 22 domains (20 PDDL + 2 YAML) |
| 3 | No long-horizon tasks | **Fixed**: IPC benchmarks include problems requiring 50-100+ step plans |
| 9 | God object PDDLEnv | Partially addressed: validator.py and agentic_env.py extract key responsibilities |

---

## Open Issues

### HIGH Priority

| # | Problem | Impact | Fix Effort |
|---|---------|--------|------------|
| 4 | No initial state randomization | Same problem = identical start state every run | Medium |
| 6 | Bare exception handlers | Silent failures in world_state.py, semantics.py, env.py | Low |
| 7 | No YAML validation | Malformed domains/tasks crash deep in execution | Medium |

### MEDIUM Priority

| # | Problem | Impact | Fix Effort |
|---|---------|--------|------------|
| 10 | No predicate indexing | O(n) fact queries slow for large worlds | Medium |
| 13 | No structured logging | Only print() in CLI | Medium |

### LOW Priority

| # | Problem | Fix Effort |
|---|---------|------------|
| 15 | Inconsistent type annotations | Low |
| 18 | Incomplete InvariantPlugin protocol | Low |

---

## Next Steps

1. **Procedural generation** for PDDL problems (parameterized difficulty, random init states)
2. **More eval baselines** across models and domains
3. **PDDLEnv refactor** to use validator.py internally (code dedup with agentic path)
4. **Fix bare exception handlers** in core modules
