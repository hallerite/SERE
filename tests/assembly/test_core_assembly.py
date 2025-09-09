import pytest
from src.io.task_loader import load_task
from src.core.world_state import lit

_ASM_TASK_YAML = """\
id: t_asm_unit_basic
name: Assembly unit tests baseline
description: Minimal assembly layout for unit tests that exercise core semantics.

meta:
  domain: assembly

objects:
  robot: [r1]
  location: [bench, cell, dock]
  container: [bin1]
  tool: [drv]
  part: [p1, p_bad, p_good]
  assembly: [asm1]
  machine: [m1]

static_facts:
  - (tool-for drv asm1)
  - (adjacent bench cell)
  - (adjacent cell bench)
  - (adjacent cell dock)
  - (adjacent dock cell)
  - (has-charger dock)

init:
  - (at r1 bench)
  - (obj-at asm1 bench)
  - (obj-at p1 bench)
  - (obj-at p_good bench)
  - (obj-at p_bad bench)
  - (obj-at bin1 bench)
  - (obj-at drv bench)
  - (clear-hand r1)

goal:
  - (installed p1 asm1)   # not used in most tests; each test asserts what it needs
"""

@pytest.fixture(scope="session")
def asm_task_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("tasks_asm")
    f = d / "t_asm_unit_basic.yaml"
    f.write_text(_ASM_TASK_YAML)
    return f

# ---- helpers -------------------------------------------------------------

def reset_with(env, *, numeric=True, conditional=True, durations=True, energy=10):
    env.enable_numeric = numeric
    env.enable_conditional = conditional
    env.enable_durations = durations
    obs, info = env.reset()
    # default energy so move works unless a test sets otherwise
    try:
        env.world.set_fluent("energy", ("r1",), energy)
    except Exception:
        pass
    return env

def M(env, frm, to): return env.step(f"<move>(move r1 {frm} {to})</move>")
def OPEN(env, c):    return env.step(f"<move>(open r1 {c})</move>")
def CLOSE(env, c):   return env.step(f"<move>(close r1 {c})</move>")
def PICK(env, o):    return env.step(f"<move>(pick-up r1 {o})</move>")
def PUT(env, o):     return env.step(f"<move>(put-down r1 {o})</move>")
def PUTIN(env, o,c): return env.step(f"<move>(put-in r1 {o} {c})</move>")
def TAKE(env,o,c):   return env.step(f"<move>(take-out r1 {o} {c})</move>")
def EQUIP(env,t):    return env.step(f"<move>(equip-tool r1 {t})</move>")
def UNEQ(env,t):     return env.step(f"<move>(unequip-tool r1 {t})</move>")
def ALIGN(env,p,a):  return env.step(f"<move>(align r1 {p} {a})</move>")
def FASTEN(env,p,a,t): return env.step(f"<move>(fasten r1 {p} {a} {t})</move>")
def UNFASTEN(env,p,a,t): return env.step(f"<move>(unfasten r1 {p} {a} {t})</move>")
def BENCH(env,o):    return env.step(f"<move>(place-on-bench r1 {o})</move>")
def RECH(env,l):     return env.step(f"<move>(recharge r1 {l})</move>")

# ---- core assembly flow --------------------------------------------------

def test_align_and_fasten_happy_path(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=30)
    reset_with(env)
    PICK(env, "drv"); EQUIP(env, "drv")
    ALIGN(env, "p1", "asm1")
    obs, r, done, info = FASTEN(env, "p1", "asm1", "drv")
    assert info.get("outcome") != "invalid"
    assert ("aligned", ("p1", "asm1")) in env.world.facts
    assert ("installed", ("p1", "asm1")) in env.world.facts
    assert ("fastened", ("p1", "asm1")) in env.world.facts

def test_fasten_on_damaged_part_does_not_install(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=30)
    reset_with(env)
    env.world.facts.add(("damaged", ("p1",)))
    PICK(env, "drv"); EQUIP(env, "drv")
    ALIGN(env, "p1", "asm1")
    obs, r, done, info = FASTEN(env, "p1", "asm1", "drv")
    assert info.get("outcome") != "invalid"
    # fasten action adds installed, but cond block deletes it when damaged
    assert ("fastened", ("p1", "asm1")) in env.world.facts
    assert ("installed", ("p1", "asm1")) not in env.world.facts

# ---- grasping / storage / inference -------------------------------------

def test_pickup_wildcard_delete_and_clear_hand(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=20)
    reset_with(env)
    obs, r, done, info = PICK(env, "p1")
    assert info.get("outcome") != "invalid"
    assert all(not (p == "obj-at" and a[0] == "p1") for p,a in env.world.facts)  # wildcard delete
    assert ("holding", ("r1", "p1")) in env.world.facts
    assert ("clear-hand", ("r1",)) not in env.world.facts

def test_putdown_infers_robot_location(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=20)
    reset_with(env)
    PICK(env, "p1")
    obs, r, done, info = PUT(env, "p1")
    assert info.get("outcome") != "invalid"
    assert ("obj-at", ("p1", "bench")) in env.world.facts
    assert ("holding", ("r1", "p1")) not in env.world.facts
    assert ("clear-hand", ("r1",)) in env.world.facts

def test_put_in_and_take_out_require_open_and_clear_hand(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=40)
    reset_with(env)
    OPEN(env, "bin1")
    PICK(env, "p1")
    obs, r, done, info = PUTIN(env, "p1", "bin1")
    assert info.get("outcome") != "invalid"
    assert ("in", ("p1", "bin1")) in env.world.facts
    assert all(not (p == "obj-at" and a[0] == "p1") for p,a in env.world.facts)
    # take-out needs clear-hand; ensure it's true
    assert ("clear-hand", ("r1",)) in env.world.facts
    obs, r, done, info = TAKE(env, "p1", "bin1")
    assert info.get("outcome") != "invalid"
    assert ("holding", ("r1", "p1")) in env.world.facts
    assert ("in", ("p1", "bin1")) not in env.world.facts

# ---- equip / unequip -----------------------------------------------------

def test_equip_and_unequip_transitions(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=30)
    reset_with(env)
    PICK(env, "drv")
    obs, r, done, info = EQUIP(env, "drv")
    assert info.get("outcome") != "invalid"
    assert ("equipped", ("r1", "drv")) in env.world.facts
    assert ("holding", ("r1", "drv")) not in env.world.facts
    assert ("clear-hand", ("r1",)) in env.world.facts
    obs, r, done, info = UNEQ(env, "drv")
    assert info.get("outcome") != "invalid"
    assert ("equipped", ("r1", "drv")) not in env.world.facts
    assert ("holding", ("r1", "drv")) in env.world.facts
    assert ("clear-hand", ("r1",)) not in env.world.facts

# ---- unfasten and bench placement ---------------------------------------

def test_unfasten_requires_colocated_and_yields_holding(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=40)
    reset_with(env)
    # pre-install p_good
    PICK(env,"drv"); EQUIP(env,"drv")
    ALIGN(env,"p_good","asm1"); FASTEN(env,"p_good","asm1","drv")
    # now unfasten requires co-location with both asm and part (domain)
    # our test task has obj-at p_good bench already -> co-located holds
    assert env.world.holds(lit("co-located","r1","p_good"))
    obs, r, done, info = UNFASTEN(env,"p_good","asm1","drv")
    assert info.get("outcome") != "invalid"
    assert ("holding", ("r1", "p_good")) in env.world.facts
    assert ("fastened", ("p_good","asm1")) not in env.world.facts
    assert ("installed", ("p_good","asm1")) not in env.world.facts
    # place it back on bench
    obs, r, done, info = BENCH(env,"p_good")
    assert info.get("outcome") != "invalid"
    assert ("obj-at", ("p_good","bench")) in env.world.facts
    assert ("clear-hand", ("r1",)) in env.world.facts

# ---- movement / adjacency / invariants ----------------------------------

def test_move_non_adjacent_fails(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=10)
    reset_with(env)
    obs, r, done, info = M(env, "bench", "dock")  # not directly adjacent per statics
    assert done and info.get("outcome") == "invalid"
    assert "adjacent" in info.get("error","")

def test_invariants_catch_multiple_locations_and_objat_in(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=10)
    reset_with(env)
    # duplicate locations
    env.world.facts.add(lit("obj-at","p1","cell"))
    with pytest.raises(ValueError) as e:
        env.reset()
    assert "multiple 'obj-at' locations" in str(e.value)
    
    # clean up the injected bad fact, then reset cleanly
    env.world.facts.discard(lit("obj-at","p1","cell"))
    reset_with(env)
    # obj-at and in conflict
    env.world.facts.add(lit("in","p1","bin1"))
    with pytest.raises(ValueError):
        env.reset()

# ---- numeric guards / durations / recharge -------------------------------

def test_move_energy_guard_and_recharge(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=50)
    reset_with(env, energy=0)
    obs, r, done, info = M(env, "bench","cell")
    assert done and info.get("outcome") == "invalid"
    # recharge at dock
    reset_with(env, energy=2)   # enough to get bench->cell->dock
    M(env,"bench","cell"); M(env,"cell","dock")
    obs, r, done, info = RECH(env, "dock")
    assert info.get("outcome") != "invalid"
    assert env.world.get_fluent("energy", ("r1",)) >= 5

def test_elapsed_increases_and_cost_accumulates(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=20)
    reset_with(env)
    t0 = env.world.get_fluent("elapsed", tuple())
    obs, r, done, info = M(env, "bench","cell")  # duration 1, cost 1
    t1 = env.world.get_fluent("elapsed", tuple())
    assert t1 >= t0 + 1 - 1e-9
    assert abs(info.get("action_cost", 0.0) - 1.0) < 1e-9

def test_wait_and_time_limit_enforced(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=10, time_limit=2.0)
    env.enable_durations = True
    reset_with(env)
    obs, r, done, info = env.step("<move>(wait 2)</move>")
    assert not done, "time == limit should not end the episode"
    obs, r, done, info = env.step("<move>(wait 0.01)</move>")
    assert done and info.get("outcome") == "loss" and info.get("reason") == "time_limit_exceeded"

# ---- derived predicate sanity --------------------------------------------

def test_co_located_truth_table_simple(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=10)
    reset_with(env)
    # r1 and asm1 at bench => co-located
    assert env.world.holds(lit("co-located","r1","asm1"))
    # move away → false
    M(env,"bench","cell")
    assert not env.world.holds(lit("co-located","r1","asm1"))
    # move back
    M(env,"cell","bench")
    assert env.world.holds(lit("co-located","r1","asm1"))
