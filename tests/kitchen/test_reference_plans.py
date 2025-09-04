from src.io.factory import load_kitchen

def test_t01_reference_plan():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=60)
    _, info = env.reset()
    plan = [
        "(move r1 hallway kitchen)",
        "(open r1 kettle1)",
        "(open r1 mug1)",
        "(toggle-kettle r1 kettle1)",
        "(toggle-kettle r1 kettle1)",
        "(toggle-kettle r1 kettle1)",
        "(move r1 kitchen pantry)",
        "(pick-up r1 teabag1)",
        "(move r1 pantry kitchen)",
        "(put-in r1 teabag1 mug1)",
        "(pour r1 kettle1 mug1)",
        "(steep-tea r1 teabag1 mug1)",
        "(move r1 kitchen table)",
    ]

    done = False
    for i, act in enumerate(plan):
        obs, r, done, info = env.step(f"<move>{act}</move>")
        outcome = info.get("outcome")
        if done:
            if outcome == "win":
                break
            else:
                raise AssertionError(
                    f"Failed at step {i}: {act}\n"
                    f"outcome={outcome}\nerror={info.get('error')}\n"
                    f"obs=\n{'observation'}"
                )

    assert info.get("outcome") == "win"

