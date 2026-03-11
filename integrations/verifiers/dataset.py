"""Task discovery and dataset building for SERE Verifiers integration."""

from __future__ import annotations

from pathlib import Path
from typing import List, Set
from importlib.resources import files as pkg_files


def discover_tasks(
    domains: List[str] | None = None,
    num_tasks_per_domain: int | None = None,
    include_multi_agent: bool = True,
    include_pddl: bool = True,
) -> List[str]:
    """
    Discover SERE task files (YAML and/or PDDL problems).

    Args:
        domains: List of domain names to include (e.g., ["kitchen", "assembly"]).
                 If None, includes all domains.
        num_tasks_per_domain: Limit number of tasks per domain.
                               If None, includes all tasks.
        include_multi_agent: Whether to include multi-agent tasks
        include_pddl: Whether to include PDDL problem files from assets/pddl/

    Returns:
        List of task paths (e.g., ["kitchen/t01_task.yaml", "pddl:blocksworld/problems/instance-1.pddl", ...])
    """
    all_tasks = _discover_yaml_tasks(domains, num_tasks_per_domain, include_multi_agent)

    if include_pddl:
        all_tasks.extend(_discover_pddl_tasks(domains, num_tasks_per_domain))

    return all_tasks


def _discover_yaml_tasks(
    domains: List[str] | None,
    num_tasks_per_domain: int | None,
    include_multi_agent: bool,
) -> List[str]:
    """Discover SERE YAML task files."""
    tasks_root = pkg_files("sere.assets.tasks")

    if not hasattr(tasks_root, "iterdir"):
        raise RuntimeError(
            "Could not access sere.assets.tasks. "
            "Please ensure SERE is properly installed."
        )

    all_tasks = []

    # Get all domain directories
    domain_dirs = [d for d in tasks_root.iterdir() if d.is_dir()]

    # Filter by requested domains
    if domains is not None:
        domain_set = set(domains)
        domain_dirs = [d for d in domain_dirs if d.name in domain_set]

    # Collect tasks from each domain
    for domain_dir in sorted(domain_dirs):
        domain_name = domain_dir.name
        domain_tasks = []

        # Find all YAML files in this domain
        for task_file in sorted(domain_dir.iterdir()):
            if task_file.suffix in {".yaml", ".yml"}:
                task_path = f"{domain_name}/{task_file.name}"

                # Filter multi-agent tasks if requested
                if not include_multi_agent:
                    # Quick check: read file and look for multi_agent: true
                    try:
                        content = task_file.read_text()
                        if "multi_agent: true" in content or "multi_agent:true" in content:
                            continue
                    except Exception:
                        # If we can't read it, include it anyway
                        pass

                domain_tasks.append(task_path)

        # Apply per-domain limit
        if num_tasks_per_domain is not None:
            domain_tasks = domain_tasks[:num_tasks_per_domain]

        all_tasks.extend(domain_tasks)

    return all_tasks


def _discover_pddl_tasks(
    domains: List[str] | None,
    num_tasks_per_domain: int | None,
) -> List[str]:
    """Discover PDDL problem files from assets/pddl/."""
    try:
        pddl_root = pkg_files("sere.assets.pddl")
    except (ModuleNotFoundError, TypeError):
        return []

    if not hasattr(pddl_root, "iterdir"):
        return []

    all_tasks = []

    # Get all domain directories that contain domain.pddl
    domain_dirs = []
    for d in pddl_root.iterdir():
        if d.is_dir():
            # Check for domain.pddl
            domain_pddl = d / "domain.pddl"
            if hasattr(domain_pddl, "is_file") and domain_pddl.is_file():
                domain_dirs.append(d)

    # Filter by requested domains
    if domains is not None:
        domain_set = set(domains)
        domain_dirs = [d for d in domain_dirs if d.name in domain_set]

    for domain_dir in sorted(domain_dirs):
        domain_name = domain_dir.name
        domain_tasks = []

        # Look for problems in problems/ subdirectory
        problems_dir = domain_dir / "problems"
        if hasattr(problems_dir, "is_dir") and problems_dir.is_dir():
            for pf in sorted(problems_dir.iterdir()):
                if str(pf.name).endswith(".pddl"):
                    # Use real filesystem path for PDDL tasks
                    domain_tasks.append(str(pf))

        # Apply per-domain limit
        if num_tasks_per_domain is not None:
            domain_tasks = domain_tasks[:num_tasks_per_domain]

        all_tasks.extend(domain_tasks)

    return all_tasks


def get_available_domains(include_pddl: bool = True) -> Set[str]:
    """
    Get list of available SERE domains.

    Returns:
        Set of domain names (e.g., {"kitchen", "assembly", "blocksworld", ...})
    """
    result = set()

    tasks_root = pkg_files("sere.assets.tasks")
    if hasattr(tasks_root, "iterdir"):
        result.update(d.name for d in tasks_root.iterdir() if d.is_dir())

    if include_pddl:
        try:
            pddl_root = pkg_files("sere.assets.pddl")
            if hasattr(pddl_root, "iterdir"):
                for d in pddl_root.iterdir():
                    if d.is_dir() and (d / "domain.pddl").is_file():
                        result.add(d.name)
        except (ModuleNotFoundError, TypeError):
            pass

    return result
