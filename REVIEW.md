# SERE Codebase Review Summary

**Review Date:** 2026-01-13
**Focus:** LLM-RL research readiness, task diversity, long-horizon capabilities

---

## Executive Summary

SERE has a **solid technical foundation** with well-designed core systems for symbolic reasoning, multi-agent coordination, and PDDL-style action execution. The codebase is **production-ready for basic planning tasks** but requires **significant expansion** in task diversity and procedural generation to become a premier LLM-RL research platform.

**Key Finding:** The primary limitation is **not** code quality or architecture—it's **task availability**. With only 20 hand-crafted tasks across 2 domains (mostly <10 steps), the environment cannot support the diverse, long-horizon training needed for modern LLM-RL research.

---

## Problems by Severity

### CRITICAL (Blocks LLM-RL Research)

| # | Problem | Impact | Location | Fix Effort |
|---|---------|--------|----------|------------|
| 1 | **No procedural task generation** | Cannot auto-generate task variants; limited to 20 hand-written tasks | Missing module | High |
| 2 | **Only 2 domains (kitchen, assembly)** | Insufficient diversity for generalization studies | `assets/domain/` | High (5+ domains) |
| 3 | **No long-horizon tasks** | Longest task is 12 steps; need 50-100 step challenges | `assets/tasks/` | Medium |
| 4 | **No initial state randomization** | Same task = identical start state every run | `task_loader.py` | Medium |
| 5 | **No combinatorial goal generation** | Cannot create multi-objective tasks programmatically | Missing module | Medium |

---

### HIGH Priority (Code Quality & Reliability)

| # | Problem | Impact | Files | Fix Effort |
|---|---------|--------|-------|------------|
| 6 | **Bare exception handlers** | Silent failures make debugging impossible | `world_state.py:80-84`<br>`semantics.py:301-305, 463-467, 562-573`<br>`env.py:268-278` | Low |
| 7 | **No YAML validation** | Malformed domains/tasks crash deep in execution | `task_loader.py`, `domain_spec.py` | Medium |
| 8 | **Poor error context** | Error messages lack action/binding info | `semantics.py:30, 96` | Low |
| 9 | **God object: PDDLEnv** | 437 lines, 39 parameters, too many responsibilities | `env.py` | High |

---

### MEDIUM Priority (Performance & Design)

| # | Problem | Impact | Files | Fix Effort |
|---|---------|--------|-------|------------|
| 10 | **No predicate indexing** | O(n) fact queries slow for large worlds | `world_state.py:46-70` | Medium |
| 11 | **Unnecessary deep copies** | Memory overhead in atomic execution | `planning.py:49-59`<br>`engine.py:359-364` | Medium |
| 12 | **Hard-coded predicate names** | "at", "holding", "in" assumed in core logic | `world_state.py:46-70`<br>`invariants.py:19-50` | Medium |
| 13 | **No structured logging** | Only `print()` in CLI; no execution traces | Entire codebase | Medium |
| 14 | **No metrics infrastructure** | Manual computation of success rate, etc. | Missing module | Medium |

---

### LOW Priority (Nice-to-Have)

| # | Problem | Files | Fix Effort |
|---|---------|-------|------------|
| 15 | Inconsistent type annotations | Mix of `List[]` vs `list[]` | Various | Low |
| 16 | Regex instead of AST | `semantics.py:10-16` | High |
| 17 | No action space introspection | Cannot query valid actions | Missing API | Medium |
| 18 | Incomplete InvariantPlugin protocol | Only `validate()` method | `invariants.py:1-7` | Low |
| 19 | No cycle detection logging | Silent in `locations_of()` | `world_state.py:28-31` | Low |
| 20 | Test coverage gaps | Malformed YAML, assembly domain | `tests/` | Medium |

---

## Natural Next Steps (Prioritized for LLM-RL)

Based on your focus (LLM-RL with Ludic, task diversity, long-horizon), here are the recommended next steps:

### Phase 1: Foundation (Month 1)
**Goal:** Enable procedural task generation

1. **Task Template System** ⭐ HIGHEST PRIORITY
   - Create `src/sere/generation/task_templates.py`
   - Support parameterized templates (beverage_type, difficulty, num_robots)
   - Auto-generate 100+ variants from single template
   - **Deliverable:** Generate 120 kitchen task variants

2. **Initial State Randomization**
   - Extend `src/sere/io/task_loader.py`
   - Randomize object positions, container contents, fluents
   - **Deliverable:** Load task with `randomize_init_state=True`

3. **Create 2 New Domains**
   - Logistics (package delivery, capacity constraints)
   - Household (cleaning, organizing, multi-room)
   - 10 tasks each (5 short, 3 medium, 2 long)
   - **Deliverable:** 20 new tasks, 2 new domain YAMLs

---

### Phase 2: Long-Horizon Tasks (Months 2-3)
**Goal:** Create challenging multi-stage tasks

4. **Combinatorial Goal Generation**
   - Create `src/sere/generation/goal_composer.py`
   - Support "clean ALL objects" and "organize by type" templates
   - **Deliverable:** Auto-generate multi-objective goals

5. **Create 20 Long-Horizon Tasks**
   - Kitchen: Restaurant shift (50+ steps)
   - Assembly: Build complex device (45+ steps)
   - Logistics: Multi-city delivery (60+ steps)
   - Household: Spring cleaning (70+ steps)
   - **Deliverable:** Tasks requiring 20-50+ actions

6. **Domain Authoring Wizard**
   - Create `src/sere/tools/domain_builder.py`
   - Interactive CLI for domain creation
   - **Deliverable:** Reduce domain authoring time from 3 days → 3 hours

---

### Phase 3: Scale Up (Months 4-7)
**Goal:** Build 500-task corpus for LLM training

7. **Task Validation Tool**
   - Create `src/sere/tools/task_validator.py`
   - Check predicates, types, reachability, reference plans
   - **Deliverable:** Validate all generated tasks

8. **Create 3 More Domains**
   - Office (document workflows)
   - Medical (patient care)
   - Construction (assembly, inspection)
   - **Deliverable:** 5 domains total

9. **Generate Full Task Corpus**
   - 500 tasks across 7 domains
   - Distribution: 200 easy, 200 medium, 80 hard, 20 very hard
   - **Deliverable:** Train/val/test splits ready for Ludic

10. **Curriculum Definitions**
    - Create `src/sere/curriculum/curriculum_spec.yaml`
    - Define progression: basics → intermediate → advanced
    - **Deliverable:** Ludic-compatible curriculum files

---

## Code Quality Fixes (As Encountered)

While implementing above, fix these high-priority issues:

### Quick Wins (Do First)
- **Fix bare exception handlers** - Replace `except Exception:` with specific types
- **Improve error messages** - Add action/binding context to ValueError messages
- **Add validation** - Check undefined predicates/types on load

### As Needed
- Add structured logging when debugging generation
- Create regression tests for procedural generation
- Document template format

---

## Success Metrics

After implementation, SERE should achieve:

| Metric | Current | Target |
|--------|---------|--------|
| **Total tasks** | 20 | 500+ |
| **Domains** | 2 | 7 |
| **Long-horizon tasks (>20 steps)** | 2 | 100+ |
| **Task generation** | Manual | Procedural |
| **State randomization** | None | Full support |
| **Domain authoring time** | 3 days | 3 hours |
| **Task authoring time** | 30 min | 5 min (via templates) |

---

## Architecture Strengths

**What SERE Does Well:**

✅ **Clean PDDL-style action grounding** - Solid symbolic reasoning foundation
✅ **Multi-agent joint actions** - True parallel execution with invariants
✅ **Ludic integration** - SereLudicEnv wrapper works well
✅ **Stochastic outcomes** - Proper RNG seeding and outcome sampling
✅ **Reward shaping** - Milestone-based with potential functions
✅ **Test coverage** - Comprehensive for core systems (1,341 lines of kitchen tests)
✅ **YAML-based authoring** - Declarative domain/task specification
✅ **Derived predicates** - Semantic feature engineering

---

## Architecture Weaknesses

**What Needs Improvement:**

❌ **Task diversity** - Only 20 hand-crafted tasks
❌ **Procedural generation** - No infrastructure for auto-generating variants
❌ **Long-horizon support** - Few tasks >15 steps
❌ **Error handling** - Bare exception handlers hide failures
❌ **Observability** - No structured logging or execution traces
❌ **Validation** - Missing YAML schema validation

---

## Specific Code Issues

### Issue #6: Bare Exception Handlers (HIGH PRIORITY)

**Location:** `world_state.py:80-84`
```python
try:
    from sere.core.semantics import eval_clause
    return bool(eval_clause(self, set(static_facts or set()), expr, {}, enable_numeric=True))
except Exception:  # ❌ TOO BROAD
    return False
```

**Problem:** Catches all exceptions indiscriminately, including:
- ImportError (if semantics module broken)
- AttributeError (if API changed)
- SyntaxError in expr
- Type errors

**Fix:**
```python
try:
    from sere.core.semantics import eval_clause
    return bool(eval_clause(...))
except (SExprError, ValueError) as e:  # ✅ SPECIFIC
    logger.warning(f"Failed to evaluate derived predicate {expr}: {e}")
    return False
```

**Also occurs in:**
- `semantics.py:301-305, 463-467, 562-573`
- `env.py:268-278`

---

### Issue #7: No YAML Validation (HIGH PRIORITY)

**Location:** `task_loader.py`, `domain_spec.py`

**Problem:** Malformed YAMLs crash deep in execution with cryptic errors.

**Examples of undetected errors:**
```yaml
# ❌ Missing 'params' field - only detected when action executed
actions:
  - name: move
    pre: ["(at ?r ?from)"]  # ?r undefined!

# ❌ Circular derived predicates - causes infinite recursion
derived:
  - pred: derived1
    when: ["(derived2)"]
  - pred: derived2
    when: ["(derived1)"]

# ❌ Undefined predicate - only fails at step()
termination:
  - when: "(undefined-pred obj1)"
```

**Fix:** Add Pydantic schemas or JSON schema validation

---

### Issue #9: God Object PDDLEnv (MEDIUM PRIORITY)

**Location:** `env.py:17-437` (437 lines!)

**Responsibilities (too many):**
- World state management
- Action execution
- Reward shaping
- Termination checking
- Message management
- Energy physics
- System prompt caching
- Formatter configuration

**Refactor into:**
1. `WorldManager` - state tracking, invariants
2. `ActionExecutor` - grounding, effects
3. `RewardShaper` - milestones, potential-based
4. `TerminationChecker` - rule evaluation
5. `PDDLEnv` (slim) - orchestrate components

---

## Recommended Reading Order

If implementing the plan, read these files first:

### Core Understanding
1. `src/sere/io/task_loader.py` - How tasks are loaded and parsed
2. `src/sere/pddl/domain_spec.py` - Domain structure
3. `src/sere/core/world_state.py` - State representation
4. `src/sere/core/pddl_env/engine.py` - Action execution

### For Task Generation
5. `src/sere/assets/domain/kitchen.yaml` - Example domain
6. `src/sere/assets/tasks/kitchen/t11_multi_agent_parallel_brew.yaml` - Complex task example
7. `task_loader.py:350-483` - Clutter injection (similar to randomization)

### For New Domains
8. `src/sere/assets/domain/assembly.yaml` - Simpler domain example
9. `src/sere/core/invariants.py` - Domain-specific constraints
10. `tests/test_domain_contracts.py` - What new domains must satisfy

---

## Conclusion

SERE is a **well-engineered symbolic reasoning platform** with solid fundamentals but **limited task diversity**. The path forward is clear:

1. **Build procedural generation** (Phase 1)
2. **Create new domains** (Phase 2)
3. **Scale to 500 tasks** (Phase 3)

This will transform SERE from a "proof of concept" into a **world-class LLM-RL research environment** suitable for:
- Curriculum learning
- Generalization studies
- Long-horizon credit assignment
- Multi-agent coordination at scale

**Next Action:** Start with task template system (highest ROI)
