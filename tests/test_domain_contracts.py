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

def _iter_domain_yaml_paths():
    """
    Yield filesystem Paths to every domain YAML packaged under:
      - sere.assets.domains/*
      - sere.assets.domain/*
    (We try plural then singular for compatibility.)
    """
    pkgs = ["sere.assets.domains", "sere.assets.domain"]
    seen = set()

    for pkg in pkgs:
        try:
            base = pkg_files(pkg)
        except Exception:
            continue
        if not base.is_dir():
            continue

        for p in base.iterdir():
            if not p.is_file():
                continue
            if not (p.name.endswith(".yaml") or p.name.endswith(".yml")):
                continue
            # dedupe across plural/singular packages if both exist
            key = (p.name, pkg)
            if key in seen:
                continue
            seen.add(key)
            with as_file(p) as real:
                yield Path(real)


ALL_DOMAIN_FILES = list(_iter_domain_yaml_paths())


def _load_domain(path: Path) -> DomainSpec:
    return DomainSpec.from_yaml(str(path))


# -------------------------
# Existing contract tests
# -------------------------

@pytest.mark.parametrize("dom_path", ALL_DOMAIN_FILES, ids=lambda p: p.name)
def test_every_action_has_success_and_a_non_success_outcome(dom_path: Path):
    """
    Contract: Each action in every domain must define outcome branches and
    include BOTH:
      - at least one branch with status 'success' (case-insensitive), and
      - at least one branch with a different status (e.g., 'fail', 'spill', etc.).
    This supports deterministic runs (enable_stochastic=False) where we prefer
    the 'success' branch if multiple are valid, but still have a non-success path.
    """
    dom = _load_domain(dom_path)

    assert dom.actions, f"{dom_path.name}: domain has no actions"

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
        f"{dom_path.name}: action outcome contract violations:\n  - " +
        "\n  - ".join(offenders)
    )


@pytest.mark.parametrize("dom_path", ALL_DOMAIN_FILES, ids=lambda p: p.name)
def test_outcome_probabilities_present_when_stochastic(dom_path: Path):
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
            f"{dom_path.name}: outcomes missing probability 'p': {', '.join(missing)}"
        )


def _has_nl(a) -> bool:
    """
    Accept either a single string or a list of strings for ActionSpec.nl.
    """
    nl = getattr(a, "nl", None)
    if isinstance(nl, str):
        return bool(nl.strip())
    if isinstance(nl, list):
        return any(isinstance(s, str) and s.strip() for s in nl)
    return False


@pytest.mark.parametrize("dom_path", ALL_DOMAIN_FILES, ids=lambda p: p.name)
def test_action_nl_present(dom_path: Path):
    """
    Ergonomics: every action should have an NL description (string or list of strings).
    """
    dom = _load_domain(dom_path)
    bad = [name for name, a in dom.actions.items() if not _has_nl(a)]
    assert not bad, f"{dom_path.name}: actions missing NL description: {bad}"


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


@pytest.mark.parametrize("dom_path", ALL_DOMAIN_FILES, ids=lambda p: p.name)
def test_outcome_when_guards_are_mutually_exclusive(dom_path: Path):
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
    assert dom.actions, f"{dom_path.name}: domain has no actions"

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
            pytest.xfail(f"{dom_path.name}:{aname}: too many guard atoms ({len(atoms)}) for brute-force exclusivity check")

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
                    f"{dom_path.name}:{aname} → overlapping outcome guards under {pretty_assign}: {valid}"
                )

    assert not violations, (
        "Outcome mutual-exclusivity violations (boolean-only check):\n  - " +
        "\n  - ".join(violations)
    )
