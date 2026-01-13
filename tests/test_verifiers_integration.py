"""Tests for SERE Verifiers integration."""

import pytest

# Import will fail if verifiers not installed - that's expected
try:
    import verifiers as vf
    from integrations.verifiers import (
        load_environment,
        SereGymWrapper,
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
    assert all(t.endswith(('.yaml', '.yml')) for t in all_tasks)

    # Discover kitchen tasks only
    kitchen_tasks = discover_tasks(domains=["kitchen"])
    assert len(kitchen_tasks) > 0
    assert all("kitchen/" in t for t in kitchen_tasks)


def test_get_available_domains():
    """Test domain listing."""
    domains = get_available_domains()
    assert len(domains) > 0
    assert "kitchen" in domains  # Should have at least kitchen domain


def test_parse_pddl_actions_single():
    """Test PDDL action parsing for single action."""
    # Raw action
    action = parse_pddl_actions("(move r1 kitchen pantry)")
    assert action == "(move r1 kitchen pantry)"

    # With explanation
    action = parse_pddl_actions("I'll move to the pantry. (move r1 kitchen pantry)")
    assert action == "(move r1 kitchen pantry)"

    # Invalid action should raise
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


def test_sere_gym_wrapper():
    """Test SereGymWrapper basic functionality."""
    wrapper = SereGymWrapper(
        task_paths=["kitchen/t01_one_step_steep.yaml"],
        env_kwargs={},
    )

    # Test reset
    obs, info = wrapper.reset(seed=0)
    assert isinstance(obs, str)
    assert len(obs) > 0
    assert "task_path" in info
    assert info["task_path"] == "kitchen/t01_one_step_steep.yaml"

    # Test step (use a valid action for the task)
    # For one-step steep, we need to steep tea
    obs, reward, done, info = wrapper.step("(steep-tea r1 leaf mug)")
    assert isinstance(obs, str)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    # Clean up
    wrapper.close()


def test_load_environment_basic():
    """Test load_environment with minimal config."""
    env = load_environment(
        task_paths=["kitchen/t01_one_step_steep.yaml"],
        episodes_per_task=1,
        eval_episodes_per_task=1,
    )

    assert env is not None
    assert hasattr(env, "dataset")
    assert hasattr(env, "eval_dataset")
    assert len(env.dataset) >= 1


def test_load_environment_domains():
    """Test load_environment with domain filtering."""
    env = load_environment(
        domains=["kitchen"],
        num_tasks_per_domain=2,
        episodes_per_task=1,
    )

    assert env is not None
    assert len(env.dataset) >= 2  # At least 2 tasks × 1 episode


def test_load_environment_multi_agent():
    """Test load_environment includes multi-agent tasks."""
    # Include multi-agent
    env_with_multi = load_environment(
        domains=["kitchen"],
        include_multi_agent=True,
        episodes_per_task=1,
    )

    # Exclude multi-agent
    env_without_multi = load_environment(
        domains=["kitchen"],
        include_multi_agent=False,
        episodes_per_task=1,
    )

    # With multi-agent should have more or equal tasks
    assert len(env_with_multi.dataset) >= len(env_without_multi.dataset)


def test_wrapper_multiple_episodes():
    """Test wrapper handles multiple episodes per task."""
    wrapper = SereGymWrapper(
        task_paths=["kitchen/t01_one_step_steep.yaml"],
        env_kwargs={},
    )

    # Episode 1
    obs1, info1 = wrapper.reset(seed=0)
    assert info1["task_path"] == "kitchen/t01_one_step_steep.yaml"

    # Episode 2 (same task, different seed)
    obs2, info2 = wrapper.reset(seed=1)
    assert info2["task_path"] == "kitchen/t01_one_step_steep.yaml"
    # Observations might be same (deterministic init) or different (stochastic)

    wrapper.close()


def test_wrapper_multiple_tasks():
    """Test wrapper cycles through multiple tasks."""
    wrapper = SereGymWrapper(
        task_paths=[
            "kitchen/t01_one_step_steep.yaml",
            "kitchen/t02_pour_then_steep.yaml",
        ],
        env_kwargs={},
    )

    # Seed 0 → task 0
    obs1, info1 = wrapper.reset(seed=0)
    assert info1["task_path"] == "kitchen/t01_one_step_steep.yaml"

    # Seed 1 → task 1
    obs2, info2 = wrapper.reset(seed=1)
    assert info2["task_path"] == "kitchen/t02_pour_then_steep.yaml"

    # Seed 2 → task 0 again (wraparound)
    obs3, info3 = wrapper.reset(seed=2)
    assert info3["task_path"] == "kitchen/t01_one_step_steep.yaml"

    wrapper.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
