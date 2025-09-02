from src.io.factory import load_kitchen

def test_t01_reference_plan():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=60)
    obs, info = env.reset()
    plan = [
        "(move r1 hallway kitchen)",
        "(open r1 kettle1 kitchen)",
        "(open r1 mug1 kitchen)",
        "(toggle-kettle r1 kettle1 kitchen)",
        "(move r1 kitchen pantry)",
        "(pick-up r1 teabag1 pantry)",
        "(move r1 pantry kitchen)",
        "(put-in r1 teabag1 mug1 kitchen)",
        "(pour r1 kettle1 mug1 kitchen)",
        "(steep-tea r1 teabag1 mug1 kitchen)",
        "(move r1 kitchen table)",
    ]
    done = False
    for act in plan:
        _, _, done, info = env.step(f"<move>{act}</move>")
        if done: break
    assert done and info.get("outcome") == "win"
