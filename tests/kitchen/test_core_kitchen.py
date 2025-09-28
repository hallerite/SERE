import pytest
from sere.io.task_loader import load_task
from sere.core.world_state import lit

_BASIC_TASK_YAML = """\
id: t01_make_tea_basic
name: Make tea and end at table
description: Basic kitchen layout for unit tests.

meta:
  domain: kitchen

objects:
  r1: robot
  hallway: location
  kitchen: location
  pantry: location
  table: location
  kettle1:
    types: [container, appliance]
  mug1:
    types: [container]
  sink1:
    types: [container]
  teabag1: object


static_facts:
  - (adjacent hallway kitchen)
  - (adjacent kitchen hallway)
  - (adjacent kitchen pantry)
  - (adjacent pantry kitchen)
  - (adjacent kitchen table)
  - (adjacent table kitchen)
  - (openable kettle1)
  - (openable mug1)
  - (is-sink sink1)
  - (pour-in-needs-open mug1)
  - (pour-out-needs-closed kettle1)

init:
  - (at r1 hallway)
  - (obj-at kettle1 kitchen)
  - (powered kettle1)
  - (obj-at mug1 kitchen)
  - (obj-at sink1 kitchen)
  - (obj-at teabag1 pantry)
  - (clear-hand r1)

# (Optional) not used by the engine in these tests, but harmless to keep
goal:
  - (tea-ready mug1)
  - (at r1 table)
"""

@pytest.fixture(scope="session")
def basic_task_file(tmp_path_factory):
    """Create a temp YAML task file once per test session."""
    d = tmp_path_factory.mktemp("tasks")
    f = d / "t01_make_tea_basic.yaml"
    f.write_text(_BASIC_TASK_YAML)
    return f

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def reset_with(env, extra_statics=None, *, numeric=True, conditional=True, energy=10):
    # enable features you’re testing
    env.enable_numeric = numeric
    env.enable_conditional = conditional

    # fresh episode
    obs, info = env.reset()

    # set sane defaults for fluents that guards depend on
    env.world.set_fluent("energy", ("r1",), energy)

    # if we add statics, re-validate and reapply the defaults
    if extra_statics:
        env.static_facts |= set(extra_statics)
        obs, info = env.reset()
        env.world.set_fluent("energy", ("r1",), energy)

    return env

def move(env, frm, to):
    return env.step(f"<move>(move r1 {frm} {to})</move>")

def open_(env, c):
    return env.step(f"<move>(open r1 {c})</move>")

def close_(env, c):
    return env.step(f"<move>(close r1 {c})</move>")

def pickup(env, o):
    return env.step(f"<move>(pick-up r1 {o})</move>")

def putdown(env, o):
    return env.step(f"<move>(put-down r1 {o})</move>")

def putin(env, o, c):
    return env.step(f"<move>(put-in r1 {o} {c})</move>")

def takeout(env, o, c):
    return env.step(f"<move>(take-out r1 {o} {c})</move>")

def power_on(env, a):
    return env.step(f"<move>(power-on r1 {a})</move>")

def heat(env, k, n):
    return env.step(f"<move>(heat-kettle r1 {k} {n})</move>")

def pour(env, k, m):
    return env.step(f"<move>(pour r1 {k} {m})</move>")

def steep(env, tb, m):
    return env.step(f"<move>(steep-tea r1 {tb} {m})</move>")

def fill(env, k, s):
    return env.step(f"<move>(fill-water r1 {k} {s})</move>")

def _affordances(env):
    # Ask the same generator the UI uses and RETURN it.
    aff = env.formatter.generate_affordances(
        env.world, env.static_facts, enable_numeric=env.enable_numeric
    )
    # Cheap sanity check so this never bites you again:
    assert isinstance(aff, list), f"affordances should be list, got {type(aff)}"
    return aff


# --------------------------------------------------------------------------------------
# Core semantics: derived preds, wildcard deletes, add inference
# --------------------------------------------------------------------------------------

def test_co_located_positive_precondition_works(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    # Co-location false in hallway
    obs, r, done, info = open_(env, "kettle1")
    assert done and info.get("outcome") == "invalid_move" and "co-located" in info.get("error", "")
    # Move to kitchen -> co-location true
    reset_with(env)
    move(env, "hallway", "kitchen")
    obs, r, done, info = open_(env, "kettle1")
    assert info.get("outcome") != "invalid"
    assert ("open", ("kettle1",)) in env.world.facts

def test_pickup_wildcard_delete_and_clearhand(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")
    obs, r, done, info = pickup(env, "mug1")
    assert info.get("outcome") != "invalid"
    # mug1 should have no obj-at fact left (wildcard delete)
    assert all(not (p == "obj-at" and a[0] == "mug1") for p, a in env.world.facts)
    # clear-hand should be consumed
    assert ("clear-hand", ("r1",)) not in env.world.facts
    # holding present
    assert ("holding", ("r1", "mug1")) in env.world.facts

def test_putdown_infers_robot_location_for_add_objat(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")
    pickup(env, "mug1")
    obs, r, done, info = putdown(env, "mug1")
    assert info.get("outcome") != "invalid"
    assert ("obj-at", ("mug1", "kitchen")) in env.world.facts
    assert ("holding", ("r1", "mug1")) not in env.world.facts
    assert ("clear-hand", ("r1",)) in env.world.facts

def test_put_in_deletes_prior_objat_and_sets_in(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    move(env, "kitchen", "pantry")
    pickup(env, "teabag1")
    move(env, "pantry", "kitchen")
    obs, r, done, info = putin(env, "teabag1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("in", ("teabag1", "mug1")) in env.world.facts
    assert all(not (p == "obj-at" and a[0] == "teabag1") for p, a in env.world.facts)
    assert ("holding", ("r1", "teabag1")) not in env.world.facts

def test_take_out_needs_open_and_clear_hand_and_works(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    move(env, "kitchen", "pantry")
    pickup(env, "teabag1")
    move(env, "pantry", "kitchen")
    putin(env, "teabag1", "mug1")
    # Now take out → should end with holding(teabag1)
    obs, r, done, info = takeout(env, "teabag1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("holding", ("r1", "teabag1")) in env.world.facts
    assert ("in", ("teabag1", "mug1")) not in env.world.facts

# --------------------------------------------------------------------------------------
# Numeric guards, conditionals, and hot water
# --------------------------------------------------------------------------------------

def test_move_energy_guard_blocks_when_insufficient(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    reset_with(env)
    env.world.set_fluent("energy", ("r1",), 0)  # move requires >= 1
    obs, r, done, info = move(env, "hallway", "kitchen")
    assert done and info.get("outcome") == "invalid_move"
    err = info.get("error", "")
    assert "preconditions were not satisfied" in err.lower()
    assert "(>= (energy r1) 1)" in err
    assert "energy(r1)=0.00" in err



def test_pour_success_branch_sets_hot_water_and_temp(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=60)
    reset_with(env)
    move(env, "hallway", "kitchen")

    # Prepare kettle with water ≥80°C
    open_(env, "kettle1")
    fill(env, "kettle1", "sink1")
    close_(env, "kettle1")
    heat(env, "kettle1", 6)   # 20 + 90 = 110°C

    open_(env, "mug1")
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    # Pour mirrors source temp now
    assert env.world.get_fluent("water-temp", ("mug1",)) == \
           pytest.approx(env.world.get_fluent("water-temp", ("kettle1",)))
    # Optional: check selected outcome branch
    assert info.get("outcome_branch") in ("transfer_hot", "chosen")


def test_pour_spill_branch_when_mug_closed(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=60)
    reset_with(env)
    move(env, "hallway", "kitchen")

    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)   # hot

    # Ensure mug is CLOSED
    env.world.facts.discard(("open", ("mug1",)))
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert ("spilled", ("mug1",)) in env.world.facts
    assert info.get("outcome_branch") in ("spill_closed_target", "chosen")

def test_steep_requires_hot_water_and_presence(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env)
    move(env, "hallway", "kitchen")

    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)   # hot enough

    open_(env, "mug1")
    pour(env, "kettle1", "mug1")

    # Insert teabag
    move(env, "kitchen", "pantry"); pickup(env, "teabag1")
    move(env, "pantry", "kitchen"); putin(env, "teabag1", "mug1")

    # Steep
    obs, r, done, info = steep(env, "teabag1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("tea-ready", ("mug1",)) in env.world.facts

# --------------------------------------------------------------------------------------
# Durations / wait / time limits
# --------------------------------------------------------------------------------------

def test_wait_and_time_limit_enforced(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=10, time_limit=2.0)
    env.enable_durations = True
    reset_with(env)
    obs, r, done, info = env.step("<move>(wait 1)</move>")
    assert not done
    obs, r, done, info = env.step("<move>(wait 2)</move>")
    assert done and info.get("outcome") == "timeout" and info.get("reason") == "time_limit_exceeded"

# --------------------------------------------------------------------------------------
# Invariants & error handling
# --------------------------------------------------------------------------------------

def test_multiple_locations_rejected_on_reset(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    reset_with(env)
    # Force illegal duplicate locations
    env.world.facts.add(lit("obj-at", "mug1", "pantry"))
    with pytest.raises(ValueError) as e:
        env.reset()
    assert "multiple 'obj-at' locations" in str(e.value)

def test_obj_at_and_in_invariant_caught_on_reset(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    reset_with(env)
    # Force illegal: both obj-at and in for same object
    env.world.facts.add(lit("in", "mug1", "kettle1"))  # nonsense on purpose
    with pytest.raises(ValueError):
        env.reset()

def test_missing_move_tags_is_invalid(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    reset_with(env)
    obs, r, done, info = env.step("(move r1 hallway kitchen)")  # missing <move> tags
    assert done and info.get("outcome") == "invalid_move"

# --------------------------------------------------------------------------------------
# Open/Close edges
# --------------------------------------------------------------------------------------

def test_open_is_guarded_and_close_requires_open(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")
    # open mug1 once
    open_(env, "mug1")
    # opening again should fail due to (not (open ?c))
    obs, r, done, info = open_(env, "mug1")
    assert done and info.get("outcome") == "invalid_move"
    # closing now should succeed (fresh episode)
    env2, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env2)
    move(env2, "hallway", "kitchen")
    open_(env2, "mug1")
    obs, r, done, info = close_(env2, "mug1")
    assert info.get("outcome") != "invalid"
    # closing again should fail
    obs, r, done, info = close_(env2, "mug1")
    assert done and info.get("outcome") == "invalid_move"

# ==================== NUMERIC SEMANTICS ====================

def test_move_decreases_energy_and_increases_time(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=10)
    reset_with(env, energy=5)
    env.enable_durations = True
    # baseline
    e0 = env.world.get_fluent("energy", ("r1",))
    t0 = env.time
    move(env, "hallway", "kitchen")
    e1 = env.world.get_fluent("energy", ("r1",))
    t1 = env.time
    assert e1 == e0 - 1, f"energy should drop by 1, got {e0}->{e1}"
    assert t1 >= t0 + 1 - 1e-9, f"time should increase by 1, got {t0}->{t1}"

def test_pour_mirrors_source_temp_not_accumulates(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")

    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)  # 110°C
    open_(env, "mug1")
    obs, r, done, info = pour(env, "kettle1", "mug1")

    assert ("has-hot-water", ("mug1",)) in env.world.facts
    # Exactly equals source temp (mirror), not add/average
    assert env.world.get_fluent("water-temp", ("mug1",)) == \
           pytest.approx(env.world.get_fluent("water-temp", ("kettle1",)))

# ==================== CONDITIONAL NO-BRANCH ====================

def test_pour_no_branch_when_temp_too_low(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")

    # Fill then under-heat: 20 + 30 = 50°C (<80)
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 2)

    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert ("has-hot-water", ("mug1",)) not in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == \
           pytest.approx(env.world.get_fluent("water-temp", ("kettle1",)))
    # New branch name: transfer_cool
    assert info.get("outcome_branch") in ("transfer_cool", "chosen")



# ==================== DERIVED AND ADJACENCY ====================

def test_co_located_truth_table_simple(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    reset_with(env)
    # At reset, r1 in hallway, kettle1 in kitchen => not co-located
    assert not env.world.holds(lit("co-located", "r1", "kettle1"))
    move(env, "hallway", "kitchen")
    assert env.world.holds(lit("co-located", "r1", "kettle1"))
    assert env.world.holds(lit("co-located", "r1", "mug1"))

def test_move_non_adjacent_fails(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    reset_with(env)
    # Assume hallway !~ pantry in your task statics
    obs, r, done, info = move(env, "hallway", "pantry")
    assert done and info.get("outcome") == "invalid_move"
    assert "adjacent" in info.get("error", "")

# ==================== EPISODE LIFECYCLE ====================

def test_invalid_step_marks_done_and_next_step_raises(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    reset_with(env)
    # Invalid: open kettle from hallway (not co-located)
    obs, r, done, info = open_(env, "kettle1")
    assert done and info.get("outcome") == "invalid_move"
    with pytest.raises(RuntimeError):
        move(env, "hallway", "kitchen")

# ==================== ADD-INFERENCE GUARDS ====================

def test_putdown_raises_when_robot_location_ambiguous(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=10)
    reset_with(env)
    move(env, "hallway", "kitchen")
    # Corrupt state: two at-locations for r1
    env.world.facts.add(lit("at", "r1", "pantry"))
    pickup(env, "mug1")
    with pytest.raises(ValueError) as e:
        putdown(env, "mug1")
    assert "infer location" in str(e.value).lower()

# ==================== CLEAR-HAND ENFORCEMENT ====================

def test_take_out_requires_clear_hand(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    # Make robot NOT clear-hand by holding mug1
    pickup(env, "mug1")
    # Also put teabag into mug (need free hand first)
    putdown(env, "mug1")               # free hand again
    move(env, "kitchen", "pantry"); pickup(env, "teabag1")
    move(env, "pantry", "kitchen"); putin(env, "teabag1", "mug1")
    # Now pick mug again to lose clear-hand
    pickup(env, "mug1")
    obs, r, done, info = takeout(env, "teabag1", "mug1")
    assert done and info.get("outcome") == "invalid_move"
    assert "clear-hand" in info.get("error","")

# ==================== TIME LIMIT EDGE ====================

def test_time_limit_boundary_exact_ok_exceed_bad(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=10, time_limit=2.0)
    env.enable_durations = True
    reset_with(env)
    obs, r, done, info = env.step("<move>(wait 2)</move>")
    assert not done, "time == limit should not end the episode"
    obs, r, done, info = env.step("<move>(wait 0.01)</move>")
    assert done and info.get("outcome") == "timeout"

# ==================== TEST "OR" ====================
# ==================== OR PRECONDITIONS / needs-open GUARD ====================

def test_pour_invalid_when_source_needs_closed_and_open(basic_task_file):
    """
    If the SOURCE container is marked (pour-out-needs-closed ?k) and it's OPEN,
    'pour' should be blocked by the (or ...) preconditions.
    Keeping the kettle CLOSED then allows pour (assuming ≥80°C).
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    # Source must be closed to pour
    reset_with(env, extra_statics=[lit("pour-out-needs-closed", "kettle1")])

    move(env, "hallway", "kitchen")

    # Prepare hot source water (fill while open, then close to heat)
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)  # ≥80°C
    open_(env, "mug1")

    # Now OPEN the kettle to violate the 'needs-closed' rule → invalid
    open_(env, "kettle1")
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert done and info.get("outcome") == "invalid_move"
    err = info.get("error", "")
    assert "preconditions were not satisfied" in err.lower()
    assert "one of:" in err
    assert "(not (pour-out-needs-closed kettle1))" in err
    assert "(not (open kettle1))" in err

    # Fresh episode: keep kettle CLOSED at pour time → success
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env, extra_statics=[lit("pour-out-needs-closed", "kettle1")])
    move(env, "hallway", "kitchen")
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)
    open_(env, "mug1")
    # Kettle remains CLOSED to satisfy 'needs-closed'
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == \
           pytest.approx(env.world.get_fluent("water-temp", ("kettle1",)))



def test_pour_spills_when_target_needs_open_and_closed(basic_task_file):
    """
    Spill occurs only if the TARGET is (pour-in-needs-open ?m) and CLOSED at pour time.
    Source openness is irrelevant for the spill decision, but source must have water.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    # Target requires being open to receive liquid
    reset_with(env, extra_statics=[lit("pour-in-needs-open", "mug1")])

    move(env, "hallway", "kitchen")

    # Prepare hot source water:
    # fill requires kettle OPEN, heat requires it CLOSED (and our domain also allows pouring while CLOSED)
    open_(env, "kettle1")
    fill(env, "kettle1", "sink1")
    close_(env, "kettle1")
    heat(env, "kettle1", 6)  # hot enough

    # Ensure mug is CLOSED to trigger spill on pour-in-needs-open
    env.world.facts.discard(("open", ("mug1",)))

    # Pour → should spill (pour-in-needs-open(mug1) ∧ not open(mug1))
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("spilled", ("mug1",)) in env.world.facts

# ==================== NUMERIC RHS & DURATION-VAR SEMANTICS ====================

def test_heat_numeric_rhs_and_duration(basic_task_file):
    """
    heat-kettle ?n:
      - increases water-temp by 15*?n
      - increases time by 0.5*?n
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    env.enable_durations = True

    move(env, "hallway", "kitchen")
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")

    # Kettle has 20°C now; heat for n=2 → 50°C; time +1.0
    obs, r, done, info = heat(env, "kettle1", 2)
    assert info.get("outcome") != "invalid"
    assert env.world.get_fluent("water-temp", ("kettle1",)) == pytest.approx(50.0)
    assert env.time >= 1.0 - 1e-9


def test_heat_multiple_ns_accumulate_linearly(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    env.enable_durations = True
    move(env, "hallway", "kitchen")

    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")

    heat(env, "kettle1", 1)  # 20 + 15 = 35°C
    obs, r, done, info = heat(env, "kettle1", 6)  # +90 → 125°C
    assert info.get("outcome") != "invalid"
    assert env.world.get_fluent("water-temp", ("kettle1",)) == pytest.approx(125.0)
    assert env.time >= 3.5 - 1e-9


def test_heat_invalid_when_open_or_unpowered(basic_task_file):
    """
    heat-kettle requires co-location, powered, has-water, and NOT open.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")

    # Opening kettle then trying to heat -> invalid due to (not (open kettle1))
    open_(env, "kettle1"); fill(env, "kettle1", "sink1")  # has-water true, still open
    obs, r, done, info = heat(env, "kettle1", 1)
    assert done and info.get("outcome") == "invalid_move"
    err = info.get("error", "")
    assert "preconditions were not satisfied" in err.lower()
    assert "(not (open kettle1))" in err
    assert "actually: true" in err.lower()

    # New episode: remove power and try to heat -> invalid due to (powered kettle1)
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    # Drop 'powered' fact
    env.world.facts.discard(("powered", ("kettle1",)))
    obs, r, done, info = heat(env, "kettle1", 1)
    assert done and info.get("outcome") == "invalid_move"
    err = info.get("error", "")
    assert "preconditions were not satisfied" in err.lower()
    assert "(powered kettle1)" in err
    assert "actually: false" in err.lower()



def test_heat_rejects_zero_or_non_numeric_n(basic_task_file):
    """
    Policy: n must be a positive number. n=0 or non-numeric token → invalid.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")

    # n = 0 → invalid
    obs, r, done, info = heat(env, "kettle1", 0)
    assert done and info.get("outcome") == "invalid_move"

    # Fresh episode for non-numeric
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")
    # Manually craft a non-numeric heat (bypass helper)
    obs, r, done, info = env.step("<move>(heat-kettle r1 kettle1 foo)</move>")
    assert done and info.get("outcome") == "invalid_move"


def test_pour_mirrors_source_temp_exactly(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")

    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)  # 110°C

    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == \
           pytest.approx(env.world.get_fluent("water-temp", ("kettle1",)))


def test_wait_duration_var_and_time_limit_via_heat(basic_task_file):
    """
    - wait r1 n advances time by n when durations are enabled.
    - a single heat that would push time over the time_limit ends the episode with loss.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20, time_limit=1.0)
    reset_with(env)
    move(env, "hallway", "kitchen")

    # Prepare kettle BEFORE turning durations on (so prep doesn't consume time)
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")

    # Enable durations now
    env.enable_durations = True

    obs, r, done, info = env.step("<move>(wait 0.4)</move>")
    assert not done
    assert env.time >= 0.4 - 1e-9

    # Now a single heat with duration 1.0 pushes time over 1.0 → loss
    obs, r, done, info = heat(env, "kettle1", 2)
    assert done and info.get("outcome") == "timeout"
    assert info.get("reason") == "time_limit_exceeded"


# ==================== SOURCE/TARGET OPENNESS GUARDS (needs-open) ====================

def test_pour_blocked_until_kettle_open_when_marked_needs_open(basic_task_file):
    """
    If the SOURCE container is marked (pour-out-needs-open ?k) and it's CLOSED,
    pour should be invalid. Opening the kettle then makes pour succeed.
    """
    # -------- Episode 1: expect INVALID while source is CLOSED --------
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env, extra_statics=[lit("pour-out-needs-open", "kettle1")])
    # Kill any conflicting default policy
    env.static_facts.discard(lit("pour-out-needs-closed", "kettle1"))

    move(env, "hallway", "kitchen")
    # Fill requires open; heat requires closed.
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)  # 20 + 90 = 110°C
    open_(env, "mug1")
    # Keep kettle CLOSED → violates needs-open constraint
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert done and info.get("outcome") == "invalid_move"
    err = info.get("error", "")
    assert "preconditions were not satisfied" in err.lower()
    assert "one of:" in err
    assert "(not (pour-out-needs-open kettle1))" in err
    assert "(open kettle1)" in err

    # -------- Episode 2: expect SUCCESS once source is OPEN --------
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env, extra_statics=[lit("pour-out-needs-open", "kettle1")])
    env.static_facts.discard(lit("pour-out-needs-closed", "kettle1"))

    move(env, "hallway", "kitchen")
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)
    open_(env, "mug1")
    open_(env, "kettle1")  # now satisfy needs-open at pour-time
    obs, r, done, info = pour(env, "kettle1", "mug1")

    assert info.get("outcome") != "invalid"
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    # target mirrors source temperature
    import pytest
    assert env.world.get_fluent("water-temp", ("mug1",)) == \
           pytest.approx(env.world.get_fluent("water-temp", ("kettle1",)))
    # optional: branch label check
    assert info.get("outcome_branch") in ("transfer_hot", "chosen")


def test_pour_spill_only_when_target_marked_needs_open_and_closed(basic_task_file):
    """
    Spill occurs only if TARGET is (needs-open ?m) AND closed at pour time.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env, extra_statics=[lit("pour-in-needs-open", "mug1")])
    move(env, "hallway", "kitchen")

    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)  # hot enough

    # Ensure mug is CLOSED
    env.world.facts.discard(("open", ("mug1",)))
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert ("spilled", ("mug1",)) in env.world.facts


# ==================== MESSAGES AFTER ACTIONS ====================

def test_action_messages_and_probes_kitchen(basic_task_file):
    """
    Verify post-action messages, including:
      - {var} and ?var substitution
      - inline probes for fluents/predicates
      - conditional-branch messages on pour (success vs. spill)
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=80)

    # --- Success path: open → fill/close → heat → pour (success) ---
    reset_with(env)
    move(env, "hallway", "kitchen")

    # 1) open mug → "{c} is now open."
    obs, r, done, info = open_(env, "mug1")
    msgs = "\n".join(info.get("messages") or [])
    assert "mug1 is now open." in msgs

    # Prepare kettle
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")

    # 2) heat kettle (n=2) → probe shows ~50°C (20 + 30)
    obs, r, done, info = heat(env, "kettle1", 2)
    msgs = "\n".join(info.get("messages") or [])
    assert "Heating kettle1" in msgs
    assert ("50.00" in msgs) or ("50.0" in msgs), f"expected temp probe ~50C in messages, got:\n{msgs}"

    # 3) heat more to reach >= 80C (n=4 → +60C)
    heat(env, "kettle1", 4)

    # 4) pour success branch → "Hot water transferred to {m}."
    obs, r, done, info = pour(env, "kettle1", "mug1")
    msgs = "\n".join(info.get("messages") or [])
    assert "Hot water transferred to mug1." in msgs

    # --- Spill branch: mug needs-open & closed ---
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env)
    move(env, "hallway", "kitchen")
    # Ensure mug is CLOSED (fixture already marks needs-open mug1)
    env.world.facts.discard(("open", ("mug1",)))
    # Prepare kettle hot again
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")
    heat(env, "kettle1", 6)
    obs, r, done, info = pour(env, "kettle1", "mug1")
    msgs = "\n".join(info.get("messages") or [])
    assert ("spill" in msgs.lower()) or ("tried to pour" in msgs.lower())

    # --- Simple brace/var substitution sanity: pick-up/put-in/close messages ---
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    move(env, "kitchen", "pantry")
    obs, r, done, info = pickup(env, "teabag1")
    msgs = "\n".join(info.get("messages") or [])
    assert "Picked up teabag1." in msgs

    move(env, "pantry", "kitchen")
    open_(env, "mug1")
    obs, r, done, info = putin(env, "teabag1", "mug1")
    msgs = "\n".join(info.get("messages") or [])
    assert "teabag1 is now in mug1." in msgs

    obs, r, done, info = close_(env, "mug1")
    msgs = "\n".join(info.get("messages") or [])
    assert "mug1 is now closed." in msgs

# ==================== DISTINCT PRECONDITIONS ====================

def test_affordances_hide_pour_with_same_source_and_target(basic_task_file):
    """
    After moving to the kitchen and preparing the source with water,
    affordances should include (pour r1 kettle1 mug1) and NEVER (mug1 -> mug1).
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")

    # Prepare source so (has-water kettle1) holds; target openness is not required for listing
    open_(env, "kettle1"); fill(env, "kettle1", "sink1"); close_(env, "kettle1")

    aff = _affordances(env)
    assert "(pour r1 kettle1 mug1)" in aff
    assert "(pour r1 mug1 mug1)" not in aff



def test_execution_rejects_same_source_and_target_with_distinct_error(basic_task_file):
    """
    Ensure all other pour preconditions are satisfied so (distinct ?k ?m)
    is the one that fails. We give mug1 water (source) to avoid failing on has-water.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")

    # Make (has-water mug1) true so only DISTINCT is the blocker
    open_(env, "mug1")
    fill(env, "mug1", "sink1")  # sets has-water(mug1) and water-temp=20

    obs, r, done, info = env.step("<move>(pour r1 mug1 mug1)</move>")
    assert done and info.get("outcome") == "invalid_move"
    err = info.get("error", "")
    assert "preconditions were not satisfied" in err.lower()
    assert "(distinct mug1 mug1)" in err
    assert "duplicates found" in err.lower()


def test_move_same_from_to_rejected_specifically_by_distinct(basic_task_file):
    """
    Make adjacency(from,to) true for kitchen->kitchen so that the ONLY failing
    guard is (distinct ?from ?to). This isolates the distinct check.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=10)
    # add adjacency(kitchen,kitchen) so adjacency won't be the blocker
    reset_with(env, extra_statics=[lit("adjacent", "kitchen", "kitchen")])

    # move to kitchen first
    move(env, "hallway", "kitchen")

    # now attempt the degenerate move
    obs, r, done, info = move(env, "kitchen", "kitchen")
    assert done and info.get("outcome") == "invalid_move"
    err = info.get("error", "")
    assert "(distinct kitchen kitchen)" in err
    assert "duplicates found" in err.lower()



def test_affordances_hide_move_same_location_even_if_adjacent_is_true(basic_task_file):
    """
    Even if adjacency(kitchen,kitchen) is true, (move r1 kitchen kitchen)
    must be suppressed by (distinct ?from ?to). Non-degenerate moves remain.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=10)
    reset_with(env, extra_statics=[lit("adjacent", "kitchen", "kitchen")])
    move(env, "hallway", "kitchen")

    aff = _affordances(env)
    assert "(move r1 kitchen kitchen)" not in aff
    assert "(move r1 kitchen hallway)" in aff

# ==================== INVALID MOVE RETRIES ====================

def test_invalid_retry_zero_terminates_immediately(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5, invalid_penalty=-0.1)
    # Hard fail policy: zero retries
    env.illegal_move_retries = 0
    reset_with(env)
    # Invalid from hallway: not co-located with kettle
    obs, r, done, info = env.step("<move>(open r1 kettle1)</move>")
    assert done and info.get("outcome") == "invalid_move"
    assert r == -0.1
    # Next step must raise
    import pytest
    with pytest.raises(RuntimeError):
        env.step("<move>(move r1 hallway kitchen)</move>")

def test_invalid_retry_one_allows_retry_without_advancing_state(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5, invalid_penalty=-0.1)
    env.illegal_move_retries = 1          # allow exactly one mulligan
    env.invalid_retry_penalty = 0.0       # make it easy to assert reward path
    reset_with(env)

    steps0 = env.steps
    time0  = env.time
    facts0 = set(env.world.facts)

    # 1st invalid → NOT done, same state, warning in obs
    obs, r, done, info = env.step("<move>(open r1 kettle1)</move>")
    assert not done
    assert "Invalid:" in obs and "Retries left" in obs
    assert env.steps == steps0, "invalid retry must not increment steps"
    assert env.time == time0, "invalid retry must not advance time"
    assert env.world.facts == facts0, "invalid retry must not mutate world"
    assert info.get("invalid_move") is True

    # Now issue a valid move → should progress and clear retry counter
    obs, r, done, info = env.step("<move>(move r1 hallway kitchen)</move>")
    assert info.get("outcome") in ("ongoing", "timeout", "success")
    # a valid transition increments steps and may change world/time
    assert env.steps == steps0 + 1
    # retry counter should clear so a future invalid gets full allowance again
    assert getattr(env, "_retries_used_this_turn", -999) == 0

def test_invalid_retry_exhausts_then_terminates(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5, invalid_penalty=-0.2)
    env.illegal_move_retries = 2   # two mulligans; the third invalid should end it
    reset_with(env)

    # 1st invalid (ok)
    obs, r, done, info = env.step("<move>(open r1 kettle1)</move>")
    assert not done
    # 2nd invalid (still ok)
    obs, r, done, info = env.step("<move>(open r1 kettle1)</move>")
    assert not done
    # 3rd invalid (exhausted -> terminal)
    obs, r, done, info = env.step("<move>(open r1 kettle1)</move>")
    assert done and info.get("outcome") == "invalid_move"
    assert r == -0.2

def test_invalid_retry_penalty_applied_on_nonterminal(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5, invalid_penalty=-1.0)
    env.illegal_move_retries = 1
    env.invalid_retry_penalty = -0.05
    reset_with(env)

    # First invalid → nonterminal, gets retry-penalty (not invalid_penalty)
    obs, r, done, info = env.step("<move>(open r1 kettle1)</move>")
    assert not done
    assert r == -0.05

    # Second invalid → terminal, gets invalid_penalty
    obs, r, done, info = env.step("<move>(open r1 kettle1)</move>")
    assert done and r == -1.0

def test_invalid_retry_keeps_observation_constant(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    env.illegal_move_retries = 1
    reset_with(env)

    # Grab baseline obs (strip the dynamic warning later)
    steps0 = env.steps
    time0 = env.time
    facts0 = set(env.world.facts)

    obs1, r, done, info = env.step("<move>(open r1 kettle1)</move>")
    assert not done
    assert env.steps == steps0
    assert env.time == time0
    assert env.world.facts == facts0
    assert obs1.lstrip().startswith("Invalid:")
    assert "Retries left" in obs1


def test_arity_mismatch_is_invalid_not_crash(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    env.illegal_move_retries = 0  # fail fast for clarity
    reset_with(env)
    # too many args for a 3-ary action
    obs, r, done, info = env.step("<move>(steep-tea r1 teabag1 mug1 kitchen)</move>")
    assert done and info.get("outcome") == "invalid_move"
    assert "Arity mismatch" in info.get("error", "")

# ==================== ILLEGAL MOVE EXPLANATIONS ====================

def test_explain_co_located_precondition(basic_task_file):
    """
    Trying to open the kettle from the hallway should:
      - include the canonical header
      - include the grounded co-located requirement
      - include the 'Actually: not co-located...' diagnostic with locations
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=10)
    env.illegal_move_retries = 0  # fail fast
    reset_with(env)  # r1@hallway, kettle1@makes kitchen

    obs, r, done, info = env.step("<move>(open r1 kettle1)</move>")
    assert done and info.get("outcome") == "invalid_move"

    err = info.get("error", "")
    # header
    assert "Preconditions were not satisfied for (open r1 kettle1):" in err
    # grounded literal + NL (the NL bit is optional, so we check the literal)
    assert "Required: (co-located r1 kettle1)" in err
    # helpful diagnostic that names the locations of x and y
    assert "Actually: not co-located." in err
    assert "x at" in err and "y at" in err


def test_explain_numeric_guard_on_move_energy(basic_task_file):
    """
    With energy=0, (move r1 hallway kitchen) should:
      - show the grounded numeric requirement (>= (energy r1) 1)
      - show the 'Actually: energy(r1)=0.00' line
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=5)
    env.illegal_move_retries = 0
    reset_with(env, numeric=True)
    env.world.set_fluent("energy", ("r1",), 0.0)

    obs, r, done, info = env.step("<move>(move r1 hallway kitchen)</move>")
    assert done and info.get("outcome") == "invalid_move"

    err = info.get("error", "")
    assert "Preconditions were not satisfied for (move r1 hallway kitchen):" in err
    assert "Required: (>= (energy r1) 1)" in err
    # the numeric renderer prints the actual current value
    assert "Actually: energy(r1)=0.00" in err or "Actually: energy(r1)=0" in err


def test_explain_distinct_guard_on_degenerate_move(basic_task_file):
    """
    Make adjacency(kitchen,kitchen) true so DISTINCT is the only blocker.
    The explanation should call out the (distinct ?from ?to) precondition.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=10)
    env.illegal_move_retries = 0
    # add adjacency to avoid that being the failure reason
    reset_with(env, extra_statics=[lit("adjacent", "kitchen", "kitchen")])

    # get to kitchen first
    obs, r, done, info = env.step("<move>(move r1 hallway kitchen)</move>")
    assert not done

    # now try the degenerate move
    obs, r, done, info = env.step("<move>(move r1 kitchen kitchen)</move>")
    assert done and info.get("outcome") == "invalid_move"

    err = info.get("error", "")
    assert "Preconditions were not satisfied for (move r1 kitchen kitchen):" in err
    err = info.get("error", "")
    assert "preconditions were not satisfied" in err.lower()
    assert "(distinct kitchen kitchen)" in err
    assert "duplicates found" in err.lower()

# ==================== TASK-LOADER / RENAME TESTS ====================

def _write_yaml(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)

def _run_plan(env, plan):
    """Run a list of raw S-exprs through the environment."""
    for a in plan:
        obs, r, done, info = env.step(f"<move>{a}</move>")
        if done and info.get("outcome") == "invalid_move":
            # bubble a helpful assertion for easier debugging
            raise AssertionError(f"Plan step failed: {a}\nerror:\n{info.get('error','')}")
    return env

def test_termination_nested_and_renamed_and_reached(tmp_path):
    """
    Kitchen task that uses placeholders + variants and a nested (and …) goal.
    We initialize energy so the reference plan's first move is valid.
    """
    yaml_text = """
id: t_loader_kitchen
name: loader_renaming_kitchen
meta:
  domain: kitchen
  enable_numeric: true
  seed: 1
  init_fluents:
    - ["energy", ["r"], 10]
objects:
  r:      {types: [robot],    variants: [r1]}
  h:      {types: [location], variants: [hallway]}
  k:      {types: [location], variants: [kitchen]}
  p:      {types: [location], variants: [pantry]}
  t:      {types: [location], variants: [table]}
  kettle: {types: [appliance, container], variants: [kettle1]}
  cup:    {types: [container], variants: [mug1]}
  tb:     {types: [object],   variants: [teabag1]}
  s:      {types: [container], variants: [sink1]}
static_facts:
  - (adjacent h k)
  - (adjacent k h)
  - (adjacent k p)
  - (adjacent p k)
  - (adjacent k t)
  - (adjacent t k)
  - (openable kettle)
  - (openable cup)
  - (powered kettle)
  - (is-sink s)
  # Optional: target pour policy; not required for the goal itself
  - (pour-in-needs-open cup)
init:
  - (at r h)
  - (obj-at kettle k)
  - (obj-at cup k)
  - (obj-at tb p)
  - (obj-at s k)
  - (clear-hand r)
termination:
  - name: goal
    when: "(and (tea-ready cup) (at r t))"
    outcome: success
reference_plan:
  - (move r h k)
  - (open r cup)
  - (open r kettle)
  - (fill-water r kettle s)
  - (close r kettle)
  - (heat-kettle r kettle 6)
  - (pour r kettle cup)
  - (move r k p)
  - (pick-up r tb)
  - (move r p k)
  - (put-in r tb cup)
  - (steep-tea r tb cup)
  - (move r k t)
"""
    p = tmp_path / "t_loader_kitchen.yaml"
    p.write_text(yaml_text)

    env, meta = load_task(None, str(p), max_steps=50)

    # placeholders should be renamed inside the nested (and …) goal
    goal_when = meta["termination"][0]["when"]
    assert "(tea-ready mug1)" in goal_when
    assert "(at r1 table)" in goal_when
    assert " cup)" not in goal_when and " r)" not in goal_when

    # --- Run the plan (seed energy explicitly after reset) ---
    obs, info = env.reset()

    for step in meta["reference_plan"]:
        obs, r, done, info = env.step(f"<move>{step}</move>")
        if done and info.get("outcome") == "invalid_move":
            raise AssertionError(f"Plan step failed: {step}\nerror:\n{info.get('error','')}")
        if done:
            break

    assert done and info.get("outcome") == "success"



def test_rename_does_not_touch_heads_kitchen(tmp_path):
    """
    Heads like 'and', 'or', '>=', 'open', 'needs-open', and 'water-temp' must remain unchanged,
    while the placeholder object 'cup' is renamed to its chosen variant.
    """
    yaml_text = """\
id: t_loader_heads_untouched
name: loader_heads_untouched
meta:
  domain: kitchen
objects:
  cup:
    types: [container]
    variants: [mug1]
static_facts: []
init: []
termination:
  - name: term
    when: "(and (or (pour-in-needs-open cup) (open cup)) (>= (water-temp cup) 80))"
    outcome: terminal
"""
    path = _write_yaml(tmp_path, "t_loader_heads_untouched.yaml", yaml_text)
    env, meta = load_task(None, path)

    when = meta["termination"][0]["when"]

    # Placeholder renamed
    assert "cup" not in when
    assert "mug1" in when

    # Heads not renamed
    for head in ("and", "or", "pour-in-needs-open", "open", ">=", "water-temp"):
        assert head in when, f"Expected head '{head}' to remain in: {when}"
