"""Task discovery and dataset building for SERE Verifiers integration."""

from __future__ import annotations

from pathlib import Path
from typing import List, Set
from importlib.resources import files as pkg_files


def discover_tasks(
    domains: List[str] | None = None,
    num_tasks_per_domain: int | None = None,
    include_multi_agent: bool = True,
) -> List[str]:
    """
    Discover SERE task YAML files.

    Args:
        domains: List of domain names to include (e.g., ["kitchen", "assembly"]).
                 If None, includes all domains.
        num_tasks_per_domain: Limit number of tasks per domain.
                               If None, includes all tasks.
        include_multi_agent: Whether to include multi-agent tasks

    Returns:
        List of task paths (e.g., ["kitchen/t01_task.yaml", ...])
    """
    tasks_root = pkg_files("sere.assets.tasks")

    if not hasattr(tasks_root, "iterdir"):
        # Fallback for older Python versions or different package structures
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


def get_available_domains() -> Set[str]:
    """
    Get list of available SERE domains.

    Returns:
        Set of domain names (e.g., {"kitchen", "assembly", ...})
    """
    tasks_root = pkg_files("sere.assets.tasks")

    if not hasattr(tasks_root, "iterdir"):
        return set()

    return {d.name for d in tasks_root.iterdir() if d.is_dir()}
