import pytest
from importlib.resources import files as pkg_files, as_file
from pathlib import Path

from sere.pddl.domain_spec import DomainSpec


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
# Tests
# -------------------------

@pytest.mark.parametrize("dom_path", ALL_DOMAIN_FILES, ids=lambda p: p.name)
def test_every_action_has_success_and_a_non_success_outcome(dom_path: Path):
    """
    Contract: Each action in every domain must define outcome branches and
    include BOTH:
      - at least one branch literally named 'success' (case-insensitive), and
      - at least one branch with a different name (e.g., 'fail', 'spill', etc.).
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

        names = {str(getattr(oc, "name", "")).lower() for oc in outcomes}
        has_success = "success" in names
        has_non_success = any(n and n != "success" for n in names)

        if not has_success or not has_non_success:
            offenders.append(
                f"{aname}: requires both a 'success' branch and a non-success branch; "
                f"found={sorted(names)}"
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


@pytest.mark.parametrize("dom_path", ALL_DOMAIN_FILES, ids=lambda p: p.name)
def test_action_nl_present(dom_path: Path):
    """
    Ergonomics: every action should have an NL description string (used in prompts).
    """
    dom = _load_domain(dom_path)

    bad = [name for name, a in dom.actions.items() if not getattr(a, "nl", "").strip()]
    assert not bad, f"{dom_path.name}: actions missing NL description: {bad}"
