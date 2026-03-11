"""
Pure plan validator: simulates a plan against a PDDL domain/world state
and reports success or the first failing step with a diagnostic trace.

No side effects, no env object, no reward/penalty system.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from sere.pddl.domain_spec import ActionSpec, DomainSpec, Predicate
from sere.pddl.grounding import ground_literal, instantiate
from sere.core.world_state import WorldState
from sere.core.semantics import (
    EvalNode,
    apply_num_eff,
    eval_clause,
    eval_trace,
    _iter_quantifier_bindings,
)


# ---------------------------------------------------------------------------
#  Result types
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of validating + applying a single action."""
    success: bool
    action_name: str
    action_args: Tuple[str, ...]
    error: str | None = None
    failed_preconditions: List[EvalNode] = field(default_factory=list)


@dataclass
class PlanResult:
    """Result of validating an entire plan."""
    success: bool               # did the plan achieve the goal?
    goal_reached: bool          # was the goal satisfied after the last step?
    steps_executed: int         # how many actions ran successfully
    total_steps: int            # total actions in the plan
    error: str | None = None    # human-readable error
    failed_step: StepResult | None = None
    final_state: WorldState | None = None


# ---------------------------------------------------------------------------
#  Single-step validation
# ---------------------------------------------------------------------------

def _resolve_action(
    domain: DomainSpec,
    world: WorldState,
    name: str,
    args: Tuple[str, ...],
) -> Tuple[Optional[ActionSpec], Optional[Dict[str, str]], Optional[str]]:
    """Resolve and type-check an action. Returns (act, bind, error)."""
    if name not in domain.actions:
        return None, None, f"Unknown action '{name}'"

    act = domain.actions[name]

    if len(args) != len(act.params):
        return None, None, (
            f"Arity mismatch for '{name}': expected {len(act.params)}, got {len(args)}"
        )

    # Type check
    for (var, typ), arg in zip(act.params, args):
        ptyp = str(typ).lower()
        if ptyp == "number":
            try:
                float(arg)
            except ValueError:
                return None, None, f"'{name}': {var} expects a number, got '{arg}'"
            continue
        obj_types = world.objects.get(arg)
        if not obj_types:
            return None, None, f"Unknown object '{arg}' for '{name}' parameter '{var}'"
        if not any(domain.is_subtype(t, ptyp) for t in obj_types):
            return None, None, (
                f"Type mismatch for '{name}': {var} expects {ptyp}, "
                f"got '{arg}' with types {sorted(obj_types)}"
            )

    bind = {var: val for (var, _), val in zip(act.params, args)}
    return act, bind, None


def validate_step(
    domain: DomainSpec,
    world: WorldState,
    static_facts: Set[Predicate],
    action_name: str,
    action_args: Tuple[str, ...],
    *,
    enable_numeric: bool = False,
    enable_conditional: bool = False,
) -> StepResult:
    """
    Validate and apply a single grounded action to the world state (in-place).
    Returns a StepResult indicating success or failure.
    """
    act, bind, err = _resolve_action(domain, world, action_name, action_args)
    if err:
        return StepResult(
            success=False,
            action_name=action_name,
            action_args=action_args,
            error=err,
        )

    assert act is not None and bind is not None

    # Check preconditions
    derived_cache: Dict[Tuple[str, Tuple[str, ...]], bool] = {}
    failed = []
    for pre in act.pre or []:
        node = eval_trace(
            world, static_facts, pre, bind,
            enable_numeric=enable_numeric,
            derived_cache=derived_cache,
        )
        if not node.satisfied:
            failed.append(node)

    if failed:
        return StepResult(
            success=False,
            action_name=action_name,
            action_args=action_args,
            error=f"Preconditions not satisfied for ({action_name} {' '.join(action_args)})",
            failed_preconditions=failed,
        )

    # Apply effects
    _, add_list, del_list = instantiate(domain, act, action_args)
    # Expand wildcard deletes
    del_list = _expand_delete_patterns(del_list, world.facts)
    world.apply(add_list, del_list)

    # Conditional effects
    if enable_conditional and act.cond:
        for cb in act.cond:
            for b2 in _iter_conditional_binds(world, cb, bind):
                ok = all(
                    eval_clause(world, static_facts, w, b2, enable_numeric=enable_numeric)
                    for w in cb.when
                )
                if ok:
                    for a in cb.add:
                        _, litp = ground_literal(a, b2)
                        world.facts.add(litp)
                    for d in cb.delete:
                        _, litp = ground_literal(d, b2)
                        world.facts.discard(litp)
                    if enable_numeric:
                        for ne in cb.num_eff:
                            apply_num_eff(world, ne, b2, {})

    # Numeric effects
    if enable_numeric and act.num_eff:
        for ne in act.num_eff:
            apply_num_eff(world, ne, bind, {})

    return StepResult(
        success=True,
        action_name=action_name,
        action_args=action_args,
    )


# ---------------------------------------------------------------------------
#  Full plan validation
# ---------------------------------------------------------------------------

def validate_plan(
    domain: DomainSpec,
    init_world: WorldState,
    static_facts: Set[Predicate],
    goal_expr: str,
    plan: List[Tuple[str, Tuple[str, ...]]],
    *,
    enable_numeric: bool = False,
    enable_conditional: bool = False,
) -> PlanResult:
    """
    Simulate a plan from init_world (on a deep copy), return success or first failure.
    """
    world = WorldState(
        domain=init_world.domain,
        objects=dict(init_world.objects),
        facts=set(init_world.facts),
        fluents=dict(init_world.fluents),
    )
    # Deep-copy object type sets
    world.objects = {k: set(v) for k, v in init_world.objects.items()}

    for i, (name, args) in enumerate(plan):
        result = validate_step(
            domain, world, static_facts, name, args,
            enable_numeric=enable_numeric,
            enable_conditional=enable_conditional,
        )
        if not result.success:
            return PlanResult(
                success=False,
                goal_reached=False,
                steps_executed=i,
                total_steps=len(plan),
                error=result.error,
                failed_step=result,
                final_state=world,
            )

    # Check goal
    goal_reached = check_goal(world, static_facts, goal_expr, enable_numeric)

    return PlanResult(
        success=goal_reached,
        goal_reached=goal_reached,
        steps_executed=len(plan),
        total_steps=len(plan),
        error=None if goal_reached else "Plan executed successfully but goal not reached",
        final_state=world,
    )


def check_goal(
    world: WorldState,
    static_facts: Set[Predicate],
    goal_expr: str,
    enable_numeric: bool = False,
) -> bool:
    """Check if the goal expression is satisfied in the current world state."""
    return eval_clause(world, static_facts, goal_expr, {}, enable_numeric=enable_numeric)


# ---------------------------------------------------------------------------
#  Formatting helpers
# ---------------------------------------------------------------------------

def format_step_error(result: StepResult, step_index: int) -> str:
    """Format a step failure into a human-readable diagnostic."""
    lines = [f"Step {step_index + 1} failed: ({result.action_name} {' '.join(result.action_args)})"]
    if result.error:
        lines.append(f"Error: {result.error}")
    for node in result.failed_preconditions:
        lines.append(f"  - {node.expr}: {_explain_node(node)}")
    return "\n".join(lines)


def format_plan_feedback(result: PlanResult) -> str:
    """Format a PlanResult into feedback for the LLM."""
    if result.success:
        return f"Plan valid! Goal reached in {result.steps_executed} steps."

    lines = []
    if result.failed_step:
        lines.append(
            f"Plan failed at step {result.steps_executed + 1} of {result.total_steps}."
        )
        lines.append("")
        step = result.failed_step
        lines.append(f"Action: ({step.action_name} {' '.join(step.action_args)})")
        if step.error:
            lines.append(f"Error: {step.error}")
        if step.failed_preconditions:
            lines.append("Unsatisfied preconditions:")
            for node in step.failed_preconditions:
                lines.append(f"  - {node.expr}: {_explain_node(node)}")
    else:
        lines.append(
            f"All {result.steps_executed} actions executed, but goal not reached."
        )

    # Show relevant state
    if result.final_state:
        lines.append("")
        lines.append(f"State after step {result.steps_executed}:")
        for pred, args in sorted(result.final_state.facts):
            lines.append(f"  ({pred} {' '.join(args)})")
        if result.final_state.fluents:
            for (fname, fargs), fval in sorted(result.final_state.fluents.items()):
                lines.append(f"  (= ({fname} {' '.join(fargs)}) {fval})")

    return "\n".join(lines)


def _explain_node(node: EvalNode) -> str:
    """One-line explanation of a failed precondition."""
    if node.kind == "num":
        d = node.details
        return f"have ({d.get('fluent', '?')} {' '.join(d.get('args', ()))}) = {d.get('current', '?')}, need {d.get('op', '?')} {d.get('rhs', '?')}"
    if node.kind == "lit":
        return "false" if not node.satisfied else "true"
    if node.kind == "not":
        return "negation failed"
    return "false"


# ---------------------------------------------------------------------------
#  Internals
# ---------------------------------------------------------------------------

def _expand_delete_patterns(
    deletes: List[Predicate], facts: Set[Predicate]
) -> List[Predicate]:
    """Expand wildcard delete patterns (unbound ?vars match any fact)."""
    expanded: List[Predicate] = []
    for pred, argtup in deletes:
        if any(isinstance(a, str) and a.startswith("?") for a in argtup):
            for p, a in list(facts):
                if p != pred:
                    continue
                ok = True
                for i, pat in enumerate(argtup):
                    if isinstance(pat, str) and pat.startswith("?"):
                        continue
                    if i >= len(a) or pat != a[i]:
                        ok = False
                        break
                if ok:
                    expanded.append((p, a))
        else:
            expanded.append((pred, argtup))
    return expanded


def _iter_conditional_binds(world, cb, base_bind):
    """Iterate over conditional-effect bindings."""
    if getattr(cb, "forall", None):
        for qb in _iter_quantifier_bindings(world, cb.forall):
            merged = dict(base_bind)
            merged.update(qb)
            yield merged
    else:
        yield base_bind
