"""Tests for SERE Verifiers integration."""

import pytest

# Import will fail if verifiers not installed - that's expected
try:
    import verifiers as vf
    from integrations.verifiers import (
        load_environment,
        SereEnv,
        parse_pddl_actions,
        discover_tasks,
        get_available_domains,
    )
    VERIFIERS_AVAILABLE = True
except ImportError:
    VERIFIERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not VERIFIERS_AVAILABLE,
    reason="Verifiers not installed. Install with: uv sync --extra verifiers"
)


def test_discover_tasks():
    """Test task discovery."""
    # Discover all tasks
    all_tasks = discover_tasks()
    assert len(all_tasks) > 0, "Should find at least one task"
    assert all(t.endswith(('.yaml', '.yml', '.pddl')) for t in all_tasks)

    # Discover kitchen tasks only
    kitchen_tasks = discover_tasks(domains=["kitchen"])
    assert len(kitchen_tasks) > 0
    assert all("kitchen" in t for t in kitchen_tasks)


def test_get_available_domains():
    """Test domain listing."""
    domains = get_available_domains()
    assert len(domains) > 0
    assert "kitchen" in domains


def test_parse_pddl_actions_single():
    """Test PDDL action parsing for single action."""
    action = parse_pddl_actions("(move r1 kitchen pantry)")
    assert action == "(move r1 kitchen pantry)"

    action = parse_pddl_actions("I'll move to the pantry. (move r1 kitchen pantry)")
    assert action == "(move r1 kitchen pantry)"

    with pytest.raises(ValueError):
        parse_pddl_actions("no action here")


def test_parse_pddl_actions_multi():
    """Test PDDL action parsing for multi-agent."""
    text = """
    I'll coordinate the robots:
    (move r1 kitchen pantry)
    (pick-up r2 leaf)
    """
    actions = parse_pddl_actions(text)
    assert isinstance(actions, list)
    assert len(actions) == 2
    assert "(move r1 kitchen pantry)" in actions
    assert "(pick-up r2 leaf)" in actions


def test_load_environment_basic():
    """Test load_environment creates a SereEnv with correct dataset."""
    env = load_environment(
        task_paths=["kitchen/t01_one_step_steep.yaml"],
        episodes_per_task=1,
        eval_episodes_per_task=1,
    )

    assert isinstance(env, SereEnv)
    assert hasattr(env, "dataset")
    assert hasattr(env, "eval_dataset")
    assert len(env.dataset) == 1
    assert len(env.eval_dataset) == 1


def test_load_environment_domains():
    """Test load_environment with domain filtering."""
    env = load_environment(
        domains=["kitchen"],
        num_tasks_per_domain=2,
        episodes_per_task=1,
    )

    assert isinstance(env, SereEnv)
    assert len(env.dataset) >= 2


def test_load_environment_multi_agent():
    """Test load_environment includes multi-agent tasks."""
    env_with_multi = load_environment(
        domains=["kitchen"],
        include_multi_agent=True,
        episodes_per_task=1,
    )
    env_without_multi = load_environment(
        domains=["kitchen"],
        include_multi_agent=False,
        episodes_per_task=1,
    )
    assert len(env_with_multi.dataset) >= len(env_without_multi.dataset)


def test_load_environment_no_upfront_resets():
    """Test that dataset construction is fast (no env resets)."""
    import time
    t0 = time.time()
    env = load_environment(
        domains=["kitchen", "assembly", "goldminer", "blocksworld", "logistics"],
        num_tasks_per_domain=10,
        episodes_per_task=1,
        eval_episodes_per_task=0,
    )
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"Dataset construction took {elapsed:.1f}s (should be <2s)"
    assert len(env.dataset) >= 30


def test_load_environment_dataset_has_task_info():
    """Test that dataset rows contain task metadata."""
    env = load_environment(
        task_paths=["kitchen/t01_one_step_steep.yaml"],
        episodes_per_task=1,
        eval_episodes_per_task=0,
    )
    row = env.dataset[0]
    assert "info" in row
    info = row["info"]
    assert info["task_path"] == "kitchen/t01_one_step_steep.yaml"
    assert info["domain"] == "kitchen"
    assert "seed" in info


def test_load_environment_multiple_episodes():
    """Test episodes_per_task creates correct number of dataset rows."""
    env = load_environment(
        task_paths=["kitchen/t01_one_step_steep.yaml"],
        episodes_per_task=3,
        eval_episodes_per_task=0,
    )
    assert len(env.dataset) == 3
    # Each episode should have a different seed
    seeds = [env.dataset[i]["info"]["seed"] for i in range(3)]
    assert len(set(seeds)) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
