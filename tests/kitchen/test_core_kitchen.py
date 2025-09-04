import pytest
from src.io.factory import load_kitchen
from src.core.world_state import lit

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

def toggle(env, k):
    return env.step(f"<move>(toggle-kettle r1 {k})</move>")

def pour(env, k, m):
    return env.step(f"<move>(pour r1 {k} {m})</move>")

def steep(env, tb, m):
    return env.step(f"<move>(steep-tea r1 {tb} {m})</move>")

# --------------------------------------------------------------------------------------
# Core semantics: derived preds, wildcard deletes, add inference
# --------------------------------------------------------------------------------------

def test_co_located_positive_precondition_works():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=20)
    reset_with(env)
    # Co-location false in hallway
    obs, r, done, info = open_(env, "kettle1")
    assert done and info.get("outcome") == "invalid" and "co-located" in info.get("error", "")
    # Move to kitchen -> co-location true
    reset_with(env)
    move(env, "hallway", "kitchen")
    obs, r, done, info = open_(env, "kettle1")
    assert info.get("outcome") != "invalid"
    assert ("open", ("kettle1",)) in env.world.facts

def test_pickup_wildcard_delete_and_clearhand():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=20)
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

def test_putdown_infers_robot_location_for_add_objat():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")
    pickup(env, "mug1")
    obs, r, done, info = putdown(env, "mug1")
    assert info.get("outcome") != "invalid"
    assert ("obj-at", ("mug1", "kitchen")) in env.world.facts
    assert ("holding", ("r1", "mug1")) not in env.world.facts
    assert ("clear-hand", ("r1",)) in env.world.facts

def test_put_in_deletes_prior_objat_and_sets_in():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=40)
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

def test_take_out_needs_open_and_clear_hand_and_works():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=40)
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

def test_move_energy_guard_blocks_when_insufficient():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=5)
    reset_with(env)
    env.world.set_fluent("energy", ("r1",), 0)  # move requires >= 1
    obs, r, done, info = move(env, "hallway", "kitchen")
    assert done and info.get("outcome") == "invalid"
    assert "Numeric precondition failed" in info.get("error", "")

def test_pour_success_branch_sets_hot_water_and_temp():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=60)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    open_(env, "kettle1")
    # Heat to >= 80 (domain increases +30 each toggle)
    toggle(env, "kettle1")
    toggle(env, "kettle1")
    toggle(env, "kettle1")
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert info.get("outcome") != "invalid"
    assert ("has-hot-water", ("mug1",)) in env.world.facts
    assert env.world.get_fluent("water-temp", ("mug1",)) == 100.0

def test_pour_spill_branch_when_mug_closed():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=60)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "kettle1")
    # Heat enough
    for _ in range(3):
        toggle(env, "kettle1")
    # Ensure mug is closed
    env.world.facts.discard(("open", ("mug1",)))
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert ("spilled", ("mug1",)) in env.world.facts

def test_steep_requires_hot_water_and_presence():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=80)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1")
    open_(env, "kettle1")
    # Heat and pour
    for _ in range(3):
        toggle(env, "kettle1")
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

def test_wait_and_time_limit_enforced():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=10, time_limit=2.0)
    env.enable_durations = True
    reset_with(env)
    obs, r, done, info = env.step("<move>(wait 1)</move>")
    assert not done
    obs, r, done, info = env.step("<move>(wait 2)</move>")
    assert done and info.get("outcome") == "loss" and info.get("reason") == "time_limit_exceeded"

# --------------------------------------------------------------------------------------
# Invariants & error handling
# --------------------------------------------------------------------------------------

def test_multiple_locations_rejected_on_reset():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=5)
    reset_with(env)
    # Force illegal duplicate locations
    env.world.facts.add(lit("obj-at", "mug1", "pantry"))
    with pytest.raises(ValueError) as e:
        env.reset()
    assert "multiple 'obj-at' locations" in str(e.value)

def test_obj_at_and_in_invariant_caught_on_reset():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=5)
    reset_with(env)
    # Force illegal: both obj-at and in for same object
    env.world.facts.add(lit("in", "mug1", "kettle1"))  # nonsense on purpose
    with pytest.raises(ValueError):
        env.reset()

def test_missing_move_tags_is_invalid():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=5)
    reset_with(env)
    obs, r, done, info = env.step("(move r1 hallway kitchen)")  # missing <move> tags
    assert done and info.get("outcome") == "invalid"

# --------------------------------------------------------------------------------------
# Open/Close edges
# --------------------------------------------------------------------------------------

def test_open_is_guarded_and_close_requires_open():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=20)
    reset_with(env)
    move(env, "hallway", "kitchen")
    # open mug1 once
    open_(env, "mug1")
    # opening again should fail due to (not (open ?c))
    obs, r, done, info = open_(env, "mug1")
    assert done and info.get("outcome") == "invalid"
    # closing now should succeed (fresh episode)
    env2, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=20)
    reset_with(env2)
    move(env2, "hallway", "kitchen")
    open_(env2, "mug1")
    obs, r, done, info = close_(env2, "mug1")
    assert info.get("outcome") != "invalid"
    # closing again should fail
    obs, r, done, info = close_(env2, "mug1")
    assert done and info.get("outcome") == "invalid"

# ==================== NUMERIC SEMANTICS ====================

def test_move_decreases_energy_and_increases_elapsed():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=10)
    reset_with(env, energy=5)
    # baseline
    e0 = env.world.get_fluent("energy", ("r1",))
    t0 = env.world.get_fluent("elapsed", tuple())
    move(env, "hallway", "kitchen")
    e1 = env.world.get_fluent("energy", ("r1",))
    t1 = env.world.get_fluent("elapsed", tuple())
    assert e1 == e0 - 1, f"energy should drop by 1, got {e0}->{e1}"
    assert t1 >= t0 + 1 - 1e-9, f"elapsed should increase by 1, got {t0}->{t1}"

def test_pour_assign_overrides_not_accumulates():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1"); open_(env, "kettle1")
    for _ in range(3): toggle(env, "kettle1")
    pour(env, "kettle1", "mug1")
    assert env.world.get_fluent("water-temp", ("mug1",)) == 100.0

def test_action_cost_accumulates_in_info():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=5)
    reset_with(env)
    move(env, "hallway", "kitchen")            # cost 1
    obs, r, done, info = open_(env, "mug1")    # cost 0.2
    assert abs(info.get("action_cost", 0.0) - 0.2) < 1e-9, "open should report its own cost this step"

# ==================== CONDITIONAL NO-BRANCH ====================

def test_pour_no_branch_when_temp_too_low():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=40)
    reset_with(env)
    move(env, "hallway", "kitchen")
    open_(env, "mug1"); open_(env, "kettle1")
    # Only one toggle -> 30C, <80C
    toggle(env, "kettle1")
    obs, r, done, info = pour(env, "kettle1", "mug1")
    assert ("has-hot-water", ("mug1",)) not in env.world.facts
    assert ("spilled", ("mug1",)) not in env.world.facts

# ==================== DERIVED AND ADJACENCY ====================

def test_co_located_truth_table_simple():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=5)
    reset_with(env)
    # At reset, r1 in hallway, kettle1 in kitchen => not co-located
    assert not env.world.holds(lit("co-located", "r1", "kettle1"))
    move(env, "hallway", "kitchen")
    assert env.world.holds(lit("co-located", "r1", "kettle1"))
    assert env.world.holds(lit("co-located", "r1", "mug1"))

def test_move_non_adjacent_fails():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=5)
    reset_with(env)
    # Assume hallway !~ pantry in your task statics
    obs, r, done, info = move(env, "hallway", "pantry")
    assert done and info.get("outcome") == "invalid"
    assert "adjacent" in info.get("error", "")

# ==================== EPISODE LIFECYCLE ====================

def test_invalid_step_marks_done_and_next_step_raises():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=5)
    reset_with(env)
    # Invalid: open kettle from hallway (not co-located)
    obs, r, done, info = open_(env, "kettle1")
    assert done and info.get("outcome") == "invalid"
    with pytest.raises(RuntimeError):
        move(env, "hallway", "kitchen")

# ==================== ADD-INFERENCE GUARDS ====================

def test_putdown_raises_when_robot_location_ambiguous():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=10)
    reset_with(env)
    move(env, "hallway", "kitchen")
    # Corrupt state: two at-locations for r1
    env.world.facts.add(lit("at", "r1", "pantry"))
    pickup(env, "mug1")
    with pytest.raises(ValueError) as e:
        putdown(env, "mug1")
    assert "infer location" in str(e.value).lower()

# ==================== CLEAR-HAND ENFORCEMENT ====================

def test_take_out_requires_clear_hand():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=40)
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
    assert done and info.get("outcome") == "invalid"
    assert "clear-hand" in info.get("error","")

# ==================== TIME LIMIT EDGE ====================

def test_time_limit_boundary_exact_ok_exceed_bad():
    env, _ = load_kitchen("tasks/kitchen/t01_make_tea_basic.yaml", max_steps=10, time_limit=2.0)
    env.enable_durations = True
    reset_with(env)
    obs, r, done, info = env.step("<move>(wait 2)</move>")
    assert not done, "time == limit should not end the episode"
    obs, r, done, info = env.step("<move>(wait 0.01)</move>")
    assert done and info.get("outcome") == "loss"
