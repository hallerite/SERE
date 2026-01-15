import pytest

from sere.io.task_loader import load_task


def test_quantified_effects_apply_to_all(tmp_path):
    domain = {
        "domain": "qfx",
        "types": [{"name": "robot"}, {"name": "item"}],
        "predicates": [
            {"name": "marked", "args": [{"name": "i", "type": "item"}]},
        ],
        "actions": [
            {
                "name": "mark-all",
                "params": [{"r": "robot"}],
                "pre": [],
                "add": [],
                "cond": [
                    {
                        "forall": [{"i": "item"}],
                        "when": ["(not (marked ?i))"],
                        "add": ["(marked ?i)"],
                    }
                ],
            }
        ],
    }

    task = {
        "id": "qfx_t01",
        "name": "Quantified Effects Mini",
        "description": "Test quantified effects over a small item set.",
        "meta": {
            "domain": "qfx",
            "enable_conditional": True,
        },
        "objects": {
            "r1": "robot",
            "i1": "item",
            "i2": "item",
            "i3": "item",
        },
        "static_facts": [],
        "init": [],
        "termination": [
            {"name": "goal", "when": "(forall (?i - item) (marked ?i))", "outcome": "success", "reward": 1.0}
        ],
        "reference_plan": ["(mark-all r1)"],
    }

    dom_path = tmp_path / "qfx.yaml"
    task_path = tmp_path / "qfx_t01.yaml"
    dom_path.write_text(__import__("json").dumps(domain, indent=2), encoding="utf-8")
    task_path.write_text(__import__("json").dumps(task, indent=2), encoding="utf-8")

    env, _ = load_task(str(dom_path), str(task_path), enable_stochastic=False)
    env.reset()
    env.step("(mark-all r1)")

    for item in ("i1", "i2", "i3"):
        assert ("marked", (item,)) in env.world.facts
