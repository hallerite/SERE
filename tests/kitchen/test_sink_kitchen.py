import pytest
from sere.io.task_loader import load_task

_BASIC_SINK_TASK_YAML = """\
id: t00_kitchen_sink_basic
name: Kitchen with sink – water handling
description: Minimal map to test sink fill and water transfer.

meta:
  domain: kitchen
  enable_numeric: true
  enable_conditional: true
  enable_stochastic: false
  init_fluents:
    - ["energy", ["r1"], 10]

objects:
  r1: robot
  kitchen: location
  pantry: location
  sink1: { types: [container] }
  kettle1: { types: [container, appliance] }
  mug1: { types: [container] }
  teabag1: object

static_facts:
  - (adjacent kitchen pantry)
  - (adjacent pantry kitchen)
  - (openable kettle1)
  - (openable mug1)
  - (pour-in-needs-open mug1)
  - (powered kettle1)
  - (is-sink sink1)

init:
  - (at r1 kitchen)
  - (obj-at sink1 kitchen)
  - (obj-at kettle1 kitchen)
  - (obj-at mug1 kitchen)
  - (obj-at teabag1 pantry)
  - (clear-hand r1)

termination:
  - name: goal
    when: "(tea-ready mug1)"
    outcome: success
"""

@pytest.fixture(scope="session")
def sink_task_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("tasks_sink")
    f = d / "t00_kitchen_sink_basic.yaml"
    f.write_text(_BASIC_SINK_TASK_YAML)
    return f

def fill(env, k, s):
    return env.step(f"<move>(fill-water r1 {k} {s})</move>")

def test_fill_requires_open_and_sets_cold_water(sink_task_file):
    env, _ = load_task(None, str(sink_task_file), max_steps=20)
    env.reset()

    # fill while closed -> invalid
    obs, r, done, info = fill(env, "kettle1", "sink1")
    assert done and info.get("outcome") == "invalid_move"
    assert "(open kettle1)" in info.get("error","")

    # fresh
    env, _ = load_task(None, str(sink_task_file), max_steps=20)
    env.reset()

    env.step("<move>(open r1 kettle1)</move>")
    obs, r, done, info = fill(env, "kettle1", "sink1")
    assert info.get("outcome") != "invalid"
    assert ("has-water", ("kettle1",)) in env.world.facts
    assert ("has-hot-water", ("kettle1",)) not in env.world.facts
    assert env.world.get_fluent("water-temp", ("kettle1",)) == pytest.approx(20.0)

def test_fill_requires_co_location_with_sink_and_vessel(sink_task_file):
    env, _ = load_task(None, str(sink_task_file), max_steps=20)
    env.reset()

    # open in kitchen first (valid), then move away
    env.step("<move>(open r1 kettle1)</move>")
    env.step("<move>(move r1 kitchen pantry)</move>")

    # not co-located with sink or kettle -> fill invalid
    obs, r, done, info = fill(env, "kettle1", "sink1")
    assert done and info.get("outcome") == "invalid_move"
    assert "co-located" in info.get("error","").lower()

def test_cold_pour_sets_water_not_hot_and_steep_blocks(sink_task_file):
    env, _ = load_task(None, str(sink_task_file), max_steps=80)
    env.reset()

    env.step("<move>(open r1 kettle1)</move>")
    fill(env, "kettle1", "sink1")

    env.step("<move>(open r1 mug1)</move>")
    obs, r, done, info = env.step("<move>(pour r1 kettle1 mug1)</move>")
    assert info.get("outcome") != "invalid"
    assert ("has-water", ("mug1",)) in env.world.facts
    assert ("has-hot-water", ("mug1",)) not in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == pytest.approx(20.0)

    env.step("<move>(move r1 kitchen pantry)</move>")
    env.step("<move>(pick-up r1 teabag1)</move>")
    env.step("<move>(move r1 pantry kitchen)</move>")
    env.step("<move>(put-in r1 teabag1 mug1)</move>")
    obs, r, done, info = env.step("<move>(steep-tea r1 teabag1 mug1)</move>")
    assert done and info.get("outcome") == "invalid_move"
    assert ">= (water-temp mug1) 80" in info.get("error","")

def test_pour_spills_and_transfers_nothing_when_target_closed(sink_task_file):
    env, _ = load_task(None, str(sink_task_file), max_steps=60)
    env.reset()

    env.step("<move>(open r1 kettle1)</move>")
    fill(env, "kettle1", "sink1")
    env.step("<move>(close r1 kettle1)</move>")
    env.step("<move>(heat-kettle r1 kettle1 6)</move>")  # 110°C

    env.world.facts.discard(("open", ("mug1",)))  # ensure closed

    obs, r, done, info = env.step("<move>(pour r1 kettle1 mug1)</move>")
    assert info.get("outcome") != "invalid"
    assert info.get("outcome_branch") in ("spill_closed_target", "chosen")
    assert ("spilled", ("mug1",)) in env.world.facts
    assert ("has-water", ("mug1",)) not in env.world.facts
    assert ("has-hot-water", ("mug1",)) not in env.world.facts
    assert ("has-water", ("kettle1",)) not in env.world.facts
    assert ("has-hot-water", ("kettle1",)) not in env.world.facts

def test_end_to_end_fill_heat_pour_steep(sink_task_file):
    env, _ = load_task(None, str(sink_task_file), max_steps=120)
    env.reset()

    env.step("<move>(open r1 kettle1)</move>")
    fill(env, "kettle1", "sink1")
    env.step("<move>(close r1 kettle1)</move>")
    env.step("<move>(heat-kettle r1 kettle1 5)</move>")  # 95°C

    env.step("<move>(open r1 mug1)</move>")
    env.step("<move>(move r1 kitchen pantry)</move>")
    env.step("<move>(pick-up r1 teabag1)</move>")
    env.step("<move>(move r1 pantry kitchen)</move>")
    env.step("<move>(put-in r1 teabag1 mug1)</move>")

    env.step("<move>(pour r1 kettle1 mug1)</move>")
    assert ("has-water", ("mug1",)) in env.world.facts
    assert ("has-hot-water", ("mug1",)) in env.world.facts

    obs, r, done, info = env.step("<move>(steep-tea r1 teabag1 mug1)</move>")
    assert info.get("outcome") in ("ongoing", "success")
    assert ("tea-ready", ("mug1",)) in env.world.facts