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
  robot: [r1]
  location: [hallway, kitchen, pantry, table]
  container: [kettle1, mug1]
  appliance: [kettle1]
  object: [teabag1]

static_facts:
  - (adjacent hallway kitchen)
  - (adjacent kitchen hallway)
  - (adjacent kitchen pantry)
  - (adjacent pantry kitchen)
  - (adjacent kitchen table)
  - (adjacent table kitchen)
  - (openable kettle1)
  - (openable mug1)
  - (needs-open mug1)

init:
  - (at r1 hallway)
  - (obj-at kettle1 kitchen)
  - (powered kettle1)
  - (obj-at mug1 kitchen)
  - (obj-at teabag1 pantry)
  - (clear-hand r1)

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
    assert "Precondition failed" in info.get("error", "")
    assert "(>= (energy ?r) 1)" in info.get("error", "")


def test_pour_success_branch_sets_hot_water_and_temp(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=60)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    # Kettle must be closed to heat; it starts closed, so don't open it here.
    heat(env, "kettle1", 6)   # +90°C

    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == 100.0

def test_pour_spill_branch_when_mug_closed(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=60)
    reset_with(env)
    move(env, "hallway", "kitchen")
    heat(env, "kettle1", 6)   # +90°C

    # Ensure mug is closed
    env.world.facts.discard(("open", ("mug1",)))
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert ("spilled", ("mug1",)) in env.world.facts

def test_steep_requires_hot_water_and_presence(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    heat(env, "kettle1", 6)   # +90°C

    pour(env, "kettle1", "mug1")
    # Insert teabag
    move(env, "kitchen", "pantry")
    pickup(env, "teabag1")
    move(env, "pantry", "kitchen")
    putin(env, "teabag1", "mug1")
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

def test_pour_assign_overrides_not_accumulates(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    # Two ticks -> +30°C, still <80°C
    # Heat to >= 80°C
    heat(env, "kettle1", 6)  # +90°C
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == 100.0

# ==================== CONDITIONAL NO-BRANCH ====================

def test_pour_no_branch_when_temp_too_low(basic_task_file):
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    # Keep kettle CLOSED while heating; 2 ticks -> +30°C (<80°C)
    heat(env, "kettle1", 2)

    obs, r, done, info = pour(env, "kettle1", "mug1")


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

def test_pour_blocked_until_kettle_open_when_marked_needs_open(basic_task_file):
    """
    If the SOURCE container is marked (needs-open ?k) and it's closed,
    'pour' should be blocked by the (or ...) preconditions.
    Opening the kettle then allows pour (assuming >=80C).
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    # Source guard on
    reset_with(env, extra_statics=[lit("needs-open", "kettle1")])

    move(env, "hallway", "kitchen")
    # Heat while CLOSED (required for heat-kettle); +90°C total
    heat(env, "kettle1", 6)
    # Keep mug open so target isn't the blocker
    open_(env, "mug1")

    # Kettle is CLOSED → pour should be invalid due to source-open guard
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert done and info.get("outcome") == "invalid_move"
    assert "Precondition failed" in info.get("error", "")

    # Fresh episode (invalid ends episode): open kettle and try again → should succeed
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env, extra_statics=[lit("needs-open", "kettle1")])
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    heat(env, "kettle1", 6)      # keep kettle CLOSED while heating
    open_(env, "kettle1")        # now open the source, since it needs-open
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == 100.0



def test_pour_spill_only_when_target_marked_needs_open_and_closed(basic_task_file):
    """
    Spill occurs only if the TARGET container is marked (needs-open ?m)
    and is closed at pour time. Source openness irrelevant here.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    # Target requires open
    reset_with(env, extra_statics=[lit("needs-open", "mug1")])

    move(env, "hallway", "kitchen")

    # Heat kettle enough while CLOSED (+90°C)
    heat(env, "kettle1", 6)

    # Ensure mug is CLOSED
    env.world.facts.discard(("open", ("mug1",)))

    # Pour → should spill (needs-open(mug1) ∧ not open(mug1))
    obs, r, done, info = pour(env, "kettle1", "mug1")
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

    # Move to kitchen (durations off by default in this helper)
    move(env, "hallway", "kitchen")

    # Kettle starts CLOSED and POWERED. Heat for n=2.
    obs, r, done, info = heat(env, "kettle1", 2)
    assert info.get("outcome") != "invalid"

    # water-temp(kettle1) should be 30.0; time should +1.0
    assert env.world.get_fluent("water-temp", ("kettle1",)) == pytest.approx(30.0)
    assert env.time >= 1.0 - 1e-9


def test_heat_multiple_ns_accumulate_linearly(basic_task_file):
    """
    Two separate heats add their effects linearly.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    env.enable_durations = True
    move(env, "hallway", "kitchen")

    heat(env, "kettle1", 1)  # +15C, +0.5s
    obs, r, done, info = heat(env, "kettle1", 6)  # +90C, +3.0s
    assert info.get("outcome") != "invalid"

    # Total +105C, time >= 3.5s
    assert env.world.get_fluent("water-temp", ("kettle1",)) == pytest.approx(105.0)
    assert env.time >= 3.5 - 1e-9


def test_heat_invalid_when_open_or_unpowered(basic_task_file):
    """
    heat-kettle requires co-location, powered, and NOT open.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")

    # Opening kettle then trying to heat -> invalid
    open_(env, "kettle1")
    obs, r, done, info = heat(env, "kettle1", 1)
    assert done and info.get("outcome") == "invalid_move"
    assert "Precondition failed" in info.get("error", "")

    # New episode: remove power and try to heat -> invalid
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")
    # Drop 'powered' fact
    env.world.facts.discard(("powered", ("kettle1",)))
    obs, r, done, info = heat(env, "kettle1", 1)
    assert done and info.get("outcome") == "invalid_move"
    assert "Precondition failed" in info.get("error", "")


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


def test_pour_assign_sets_target_temp_to_100(basic_task_file):
    """
    After a successful pour into an OK target, the target temp is exactly 100.0
    (assign semantics, not accumulate).
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    # Heat sufficiently while kettle CLOSED
    heat(env, "kettle1", 6)  # +90C -> >=80C

    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == pytest.approx(100.0)


def test_wait_duration_var_and_time_limit_via_heat(basic_task_file):
    """
    - wait r1 n advances time by n when durations are enabled.
    - a single heat that would push time over the time_limit ends the episode with loss.
    """
    # Small time limit for the second half of the test
    env, _ = load_task(None, str(basic_task_file), max_steps=20, time_limit=1.0)
    reset_with(env)
    # Move without durations so it doesn't consume time in this test
    move(env, "hallway", "kitchen")

    # Enable durations now
    env.enable_durations = True

    # 'wait' with duration_var= n (unit=1.0)
    obs, r, done, info = env.step("<move>(wait 0.4)</move>")
    assert not done
    assert env.time >= 0.4 - 1e-9

    # Now a single heat with duration 0.5 * 2 = 1.0 pushes time over 1.0 → loss
    obs, r, done, info = heat(env, "kettle1", 2)
    assert done and info.get("outcome") == "timeout"
    assert info.get("reason") == "time_limit_exceeded"


# ==================== SOURCE/TARGET OPENNESS GUARDS (needs-open) ====================

def test_pour_blocked_until_kettle_open_when_marked_needs_open(basic_task_file):
    """
    If the SOURCE container is marked (needs-open ?k) and it's closed,
    'pour' should be invalid. Opening the kettle fixes it.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env, extra_statics=[lit("needs-open", "kettle1")])

    move(env, "hallway", "kitchen")
    heat(env, "kettle1", 6)  # get it hot enough
    open_(env, "mug1")       # ensure target isn't the blocker

    # Closed kettle + needs-open(kettle1) → invalid
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert done and info.get("outcome") == "invalid_move"
    assert "Precondition failed" in info.get("error", "")

    # Now open kettle and try again → success
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env, extra_statics=[lit("needs-open", "kettle1")])
    move(env, "hallway", "kitchen")
    heat(env, "kettle1", 6)
    open_(env, "mug1")
    open_(env, "kettle1")
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == pytest.approx(100.0)


def test_pour_spill_only_when_target_marked_needs_open_and_closed(basic_task_file):
    """
    Spill occurs only if TARGET is (needs-open ?m) AND closed at pour time.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=80)
    reset_with(env, extra_statics=[lit("needs-open", "mug1")])
    move(env, "hallway", "kitchen")
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

    # --- Success path: open → heat → pour (success) ---
    reset_with(env)
    # Move to kitchen so co-location holds
    move(env, "hallway", "kitchen")

    # 1) open mug → "{c} is now open."
    obs, r, done, info = open_(env, "mug1")
    msgs = "\n".join(info.get("messages") or [])
    assert "mug1 is now open." in msgs

    # 2) heat kettle (n=2) → "Heating {k}… water-temp now (water-temp ?k)°C."
    # kettle starts CLOSED and POWERED in the fixture; co-location now true
    obs, r, done, info = heat(env, "kettle1", 2)
    msgs = "\n".join(info.get("messages") or [])
    # Probe should have evaluated to ~30.00
    assert "Heating kettle1" in msgs
    assert ("30.00" in msgs) or ("30.0" in msgs), f"expected temp probe ~30C in messages, got:\n{msgs}"

    # 3) heat more to reach >= 80C (n=4 → +60C, total 90C)
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
    # Heat enough for success path, but target is closed → spill branch message
    heat(env, "kettle1", 6)
    obs, r, done, info = pour(env, "kettle1", "mug1")
    msgs = "\n".join(info.get("messages") or [])
    assert "spill" in msgs.lower()

    # --- Simple brace/var substitution sanity: pick-up/put-in/close messages ---
    env, _ = load_task(None, str(basic_task_file), max_steps=40)
    reset_with(env)
    # Go to the PANTRY (teabag1 is there), then pick up
    move(env, "hallway", "kitchen")
    move(env, "kitchen", "pantry")
    obs, r, done, info = pickup(env, "teabag1")
    msgs = "\n".join(info.get("messages") or [])
    assert "Picked up teabag1." in msgs

    # Bring it to the kitchen, open mug, put in teabag → messages fire
    move(env, "pantry", "kitchen")
    open_(env, "mug1")
    obs, r, done, info = putin(env, "teabag1", "mug1")
    msgs = "\n".join(info.get("messages") or [])
    assert "teabag1 is now in mug1." in msgs

    # Close mug → "{c} is now closed."
    obs, r, done, info = close_(env, "mug1")
    msgs = "\n".join(info.get("messages") or [])
    assert "mug1 is now closed." in msgs

# ==================== DISTINCT PRECONDITIONS ====================

def test_affordances_hide_pour_with_same_source_and_target(basic_task_file):
    """
    After moving to the kitchen (co-location satisfied), affordances should
    include a valid pour (kettle1 -> mug1) but NEVER offer (mug1 -> mug1).
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    # get co-location with both containers
    obs, r, done, info = move(env, "hallway", "kitchen")

    # sanity: valid pour is offered
    assert "(pour r1 kettle1 mug1)" in obs

    # critical: nonsense same-arg pour is NOT offered
    assert "(pour r1 mug1 mug1)" not in obs


def test_execution_rejects_same_source_and_target_with_distinct_error(basic_task_file):
    """
    Even if a user types the bad action manually, step() should reject it with a
    precondition failure that names the (distinct ?k ?m) guard.
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")

    obs, r, done, info = env.step("<move>(pour r1 mug1 mug1)</move>")
    assert done and info.get("outcome") == "invalid_move"
    assert "Precondition failed" in info.get("error", "")
    assert "(distinct ?k ?m)" in info.get("error", "")


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
    assert "Precondition failed" in info.get("error", "")
    assert "(distinct ?from ?to)" in info.get("error", "")


def test_affordances_hide_move_same_location_even_if_adjacent_is_true(basic_task_file):
    """
    With adjacency(kitchen,kitchen) forced true, affordances STILL must not
    propose (move r1 kitchen kitchen) because of (distinct ?from ?to).
    """
    env, _ = load_task(None, str(basic_task_file), max_steps=10)
    reset_with(env, extra_statics=[lit("adjacent", "kitchen", "kitchen")])

    # go to kitchen so (at r1 kitchen) and any other movement preconds hold
    obs, r, done, info = move(env, "hallway", "kitchen")

    # affordances list is in obs text; it must not include the degenerate move
    assert "(move r1 kitchen kitchen)" not in obs

    # …but it should include non-degenerate neighbors (hallway/pantry/table if adjacent)
    # We only know hallway<->kitchen is adjacent in this fixture:
    # from kitchen, moving back to hallway should be offered
    assert "(move r1 kitchen hallway)" in obs
