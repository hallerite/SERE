import re
import itertools
import pytest
from importlib.resources import files as pkg_files, as_file
from pathlib import Path

from sere.pddl.domain_spec import DomainSpec
from sere.core.world_state import WorldState
from sere.core.semantics import eval_clause


# -------------------------
# Utilities
# -------------------------

def _iter_pddl_domain_dirs():
    """
    Yield (name, filesystem Path) for every PDDL domain directory packaged
    under sere.assets.pddl that contains a domain.pddl file.
    """
    try:
        base = pkg_files("sere.assets.pddl")
    except Exception:
        return
    if not base.is_dir():
        return
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        domain_pddl = p / "domain.pddl"
        if domain_pddl.is_file():
            with as_file(p) as real:
                yield (p.name, Path(real))


def _load_pddl_domain_spec(domain_dir: Path) -> DomainSpec:
    from sere.io.pddl_loader import load_pddl_domain
    spec, _meta, _raw = load_pddl_domain(str(domain_dir))
    return spec


ALL_PDDL_DOMAINS = list(_iter_pddl_domain_dirs())

# Domains with extensions.yaml (NL, outcomes, etc.)
ALL_EXTENDED_DOMAINS = [
    (name, path) for name, path in ALL_PDDL_DOMAINS
    if (path / "extensions.yaml").exists()
]


def _load_domain(name_and_path) -> DomainSpec:
    _name, path = name_and_path
    return _load_pddl_domain_spec(path)


# -------------------------
# Existing contract tests
# -------------------------

@pytest.mark.parametrize("dom_path", ALL_EXTENDED_DOMAINS, ids=lambda x: x[0])
def test_every_action_has_success_and_a_non_success_outcome(dom_path):
    """
    Contract: Each action in every stochastic domain must define outcome branches
    and include BOTH:
      - at least one branch with status 'success' (case-insensitive), and
      - at least one branch with a different status (e.g., 'fail', 'spill', etc.).
    Skips domains where no action defines outcomes (pure PDDL / deterministic).
    """
    dom_name = dom_path[0]
    dom = _load_domain(dom_path)

    assert dom.actions, f"{dom_name}: domain has no actions"

    # Skip deterministic domains with no outcomes defined
    if not any(getattr(a, "outcomes", None) for a in dom.actions.values()):
        pytest.skip(f"{dom_name}: no outcomes defined (deterministic domain)")

    offenders = []
    for aname, act in sorted(dom.actions.items()):
        outcomes = list(getattr(act, "outcomes", []) or [])
        if not outcomes:
            offenders.append(f"{aname}: has no outcomes[] defined")
            continue

        missing_status = [oc for oc in outcomes if not getattr(oc, "status", None)]
        if missing_status:
            offenders.append(f"{aname}: all outcomes must define status (success/fail)")
            continue

        statuses = {str(getattr(oc, "status", "")).lower() for oc in outcomes}
        has_success = "success" in statuses
        has_non_success = any(s and s != "success" for s in statuses)

        if not has_success or not has_non_success:
            offenders.append(
                f"{aname}: requires both a 'success' branch and a non-success branch; "
                f"found={sorted(statuses)}"
            )

    assert not offenders, (
        f"{dom_name}: action outcome contract violations:\n  - " +
        "\n  - ".join(offenders)
    )


@pytest.mark.parametrize("dom_path", ALL_EXTENDED_DOMAINS, ids=lambda x: x[0])
def test_outcome_probabilities_present_when_stochastic(dom_path):
    """
    Soft guard: if an action declares outcomes, every outcome should provide a 'p'
    field (even if tests later run with enable_stochastic=False). This prevents
    accidental NaNs when sampling.
    We only warn (xfail) if a domain intentionally omits probabilities.
    """
    dom = _load_domain(dom_path)

    missing = []
    for aname, act in dom.actions.items():
        outcomes = list(getattr(act, "outcomes", []) or [])
        if not outcomes:
            continue
        for oc in outcomes:
            # Some branches might omit 'p' intentionally; collect to review.
            if getattr(oc, "p", None) is None:
                oc_name = str(getattr(oc, "name", ""))
                missing.append(f"{aname}.{oc_name}")

    if missing:
        pytest.xfail(
            f"{dom_path[0]}: outcomes missing probability 'p': {', '.join(missing)}"
        )




def test_domain_load_rejects_static_effects():
    y = {
        "domain": "dummy",
        "predicates": [{"name": "p", "args": [], "static": True}],
        "actions": [{"name": "a", "params": [], "pre": [], "add": ["(p)"], "delete": []}],
    }
    with pytest.raises(ValueError, match="Static predicate effects"):
        DomainSpec.from_dict(y)


# -------------------------
# Mutual exclusivity of outcome guards (boolean-only)
# -------------------------

_NUMERIC_HEADS = {"<", ">", "<=", ">=", "="}
_LOGICAL_HEADS = {"and", "or", "not"}
_BUILTIN_HEADS = _NUMERIC_HEADS | _LOGICAL_HEADS

def _sexpr_heads(s: str):
    """
    Extract simple atoms like (pred a b) from a string S-expression.
    Returns [(head, args_tuple), ...]. Numeric/logical heads are retained for
    structure but filtered later.
    """
    out = []
    for m in re.finditer(r"\(([^\s()]+)([^()]*)\)", s or ""):
        head = m.group(1)
        args = tuple(a for a in m.group(2).strip().split() if a)
        out.append((head, args))
    return out


def _ground_args(args, bind):
    return tuple(bind.get(a[1:], a) if a.startswith("?") else a for a in args)


def _atoms_in_when_guards(outcomes, bind):
    """
    Collect grounded predicate atoms (excluding numeric/logical heads)
    appearing in outcomes[].when after applying VAR->SYMBOL binding.
    """
    atoms = set()
    for oc in outcomes or []:
        for w in getattr(oc, "when", []) or []:
            for head, args in _sexpr_heads(str(w)):
                if head in _BUILTIN_HEADS:
                    continue
                atoms.add((head, _ground_args(args, bind)))
    return sorted(atoms)


def _all_boolean_assignments(n: int):
    for bits in itertools.product([False, True], repeat=n):
        yield bits


@pytest.mark.parametrize("dom_path", ALL_EXTENDED_DOMAINS, ids=lambda x: x[0])
def test_outcome_when_guards_are_mutually_exclusive(dom_path):
    """
    For each action with >=2 outcomes, brute-force boolean truth assignments over the
    predicate atoms referenced in their `when` guards and assert that no state
    makes two or more branches true simultaneously.

    Scope/assumptions:
      - Only boolean predicates are toggled (we ignore numeric guards).
      - If an action's guards are numeric-only (no predicate atoms), we skip it.
      - This catches overlaps like:
          spill:   (and (needs-open ?m) (not (open ?m)))
          success: (or (not (needs-open ?m)) (open ?m))
    """
    dom = _load_domain(dom_path)
    dom_name = dom_path[0]
    dom = _load_domain(dom_path)
    assert dom.actions, f"{dom_name}: domain has no actions"

    violations = []

    for aname, act in sorted(dom.actions.items()):
        outcomes = list(getattr(act, "outcomes", []) or [])
        if len(outcomes) < 2:
            continue

        # Build minimal world + symbol binding per param type
        bind = {}
        world = WorldState(dom)
        for (var, typ) in act.params:
            sym = f"{var.strip('?')}_0"
            bind[var.strip('?')] = sym
            world.add_object(sym, typ)

        atoms = _atoms_in_when_guards(outcomes, bind)
        if not atoms:
            # No boolean atoms to toggle (likely numeric-only guards) -> skip
            continue

        MAX_ATOMS = 12
        if len(atoms) > MAX_ATOMS:
            pytest.xfail(f"{dom_name}:{aname}: too many guard atoms ({len(atoms)}) for brute-force exclusivity check")

        for bits in _all_boolean_assignments(len(atoms)):
            # Install facts for this assignment
            world.facts.clear()
            for (truth, (pred, args)) in zip(bits, atoms):
                if truth:
                    world.facts.add((pred, args))

            # Evaluate branches under this assignment; ignore numeric (we don't set fluents)
            valid = []
            for oc in outcomes:
                conds = getattr(oc, "when", []) or []
                if all(eval_clause(world, set(), str(c), bind, enable_numeric=False) for c in conds):
                    valid.append(str(getattr(oc, "name", "")) or "?")

            if len(valid) > 1:
                pretty_assign = {f"({p} {' '.join(a)})": b for (p, a), b in zip(atoms, bits)}
                violations.append(
                    f"{dom_name}:{aname} → overlapping outcome guards under {pretty_assign}: {valid}"
                )

    assert not violations, (
        "Outcome mutual-exclusivity violations (boolean-only check):\n  - " +
        "\n  - ".join(violations)
    )
