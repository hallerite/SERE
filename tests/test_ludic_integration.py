from pathlib import Path
import sys


LUDIC_SRC = Path(__file__).resolve().parents[1] / "ludic" / "src"
if LUDIC_SRC.exists() and str(LUDIC_SRC) not in sys.path:
    sys.path.insert(0, str(LUDIC_SRC))

from sere.core.pddl_env.run_mode import RunMode
from sere.integrations.ludic_env import SereLudicEnv
from sere.integrations.ludic_parser import pddl_action_parser
from sere.io.task_loader import load_task


TASK_PATH = "kitchen/t11_multi_agent_parallel_brew.yaml"


def _make_multi_agent_env(*, max_steps: int | None = None) -> SereLudicEnv:
    env_kwargs = dict(run_mode=RunMode.INTERACTIVE, enable_stochastic=False)
    if max_steps is not None:
        env_kwargs["max_steps"] = max_steps
    env, _meta = load_task(None, TASK_PATH, **env_kwargs)
    return SereLudicEnv(env)


def test_pddl_action_parser_accepts_single_action() -> None:
    parser = pddl_action_parser()
    result = parser("(move r1 kitchen pantry)")
    assert result.action == "(move r1 kitchen pantry)"
    assert result.reward == 0.0
    assert result.obs is None


def test_pddl_action_parser_rejects_multiple_actions() -> None:
    parser = pddl_action_parser()
    result = parser("(a)(b)")
    assert result.action is None
    assert result.obs


def test_multi_agent_missing_action_is_invalid() -> None:
    env = _make_multi_agent_env()
    env.reset()
    assert env.agent_ids == ["r1", "r2"]

    outcomes = env.step({"r1": "(idle r1)"})
    for agent_id in env.agent_ids:
        outcome = outcomes[agent_id]
        assert outcome.info.get("outcome") == "invalid_move"
        assert outcome.terminated is True
        assert outcome.truncated is False


def test_timeout_maps_to_truncated() -> None:
    env = _make_multi_agent_env(max_steps=0)
    env.reset()

    outcomes = env.step({"r1": "(idle r1)", "r2": "(idle r2)"})
    for outcome in outcomes.values():
        assert outcome.truncated is True
        assert outcome.terminated is False
