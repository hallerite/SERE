import pytest
from sere.io.task_loader import load_task
from sere.core.world_state import lit

_ASM_TASK_YAML = """\
id: t_asm_unit_basic
name: Assembly unit tests baseline
description: Minimal assembly layout for unit tests that exercise core semantics.

meta:
  domain: assembly

objects:
  r1: robot
  bench: location
  cell: location
  dock: location
  bin1: container
  drv: tool
  p1: part
  p_bad: part
  p_good: part
  asm1: assembly
  m1: machine

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
  - (installed p1 asm1)
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

def M(env, frm, to): return env.step(f"(move r1 {frm} {to})")
def OPEN(env, c):    return env.step(f"(open r1 {c})")
def CLOSE(env, c):   return env.step(f"(close r1 {c})")
def PICK(env, o):    return env.step(f"(pick-up r1 {o})")
def PUT(env, o):     return env.step(f"(put-down r1 {o})")
def PUTIN(env, o,c): return env.step(f"(put-in r1 {o} {c})")
def TAKE(env,o,c):   return env.step(f"(take-out r1 {o} {c})")
def EQUIP(env,t):    return env.step(f"(equip-tool r1 {t})")
def UNEQ(env,t):     return env.step(f"(unequip-tool r1 {t})")
def ALIGN(env,p,a):  return env.step(f"(align r1 {p} {a})")
def FASTEN(env,p,a,t): return env.step(f"(fasten r1 {p} {a} {t})")
def UNFASTEN(env,p,a,t): return env.step(f"(unfasten r1 {p} {a} {t})")
def BENCH(env,o):    return env.step(f"(place-on-bench r1 {o})")
def RECH(env,l):     return env.step(f"(recharge r1 {l})")

# ---- core assembly flow --------------------------------------------------

def test_align_and_fasten_happy_path(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=30)
    reset_with(env)
    PICK(env, "drv"); EQUIP(env, "drv")
    ALIGN(env, "p1", "asm1")
    obs, r, done, info = FASTEN(env, "p1", "asm1", "drv")
    assert info.get("outcome") != "invalid_move"
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
    assert info.get("outcome") != "invalid_move"
    # fasten action adds installed, but cond block deletes it when damaged
    assert ("fastened", ("p1", "asm1")) in env.world.facts
    assert ("installed", ("p1", "asm1")) not in env.world.facts

# ---- grasping / storage / inference -------------------------------------

def test_pickup_wildcard_delete_and_clear_hand(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=20)
    reset_with(env)
    obs, r, done, info = PICK(env, "p1")
    assert info.get("outcome") != "invalid_move"
    assert all(not (p == "obj-at" and a[0] == "p1") for p,a in env.world.facts)  # wildcard delete
    assert ("holding", ("r1", "p1")) in env.world.facts
    assert ("clear-hand", ("r1",)) not in env.world.facts

def test_putdown_infers_robot_location(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=20)
    reset_with(env)
    PICK(env, "p1")
    obs, r, done, info = PUT(env, "p1")
    assert info.get("outcome") != "invalid_move"
    assert ("obj-at", ("p1", "bench")) in env.world.facts
    assert ("holding", ("r1", "p1")) not in env.world.facts
    assert ("clear-hand", ("r1",)) in env.world.facts

def test_put_in_and_take_out_require_open_and_clear_hand(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=40)
    reset_with(env)
    OPEN(env, "bin1")
    PICK(env, "p1")
    obs, r, done, info = PUTIN(env, "p1", "bin1")
    assert info.get("outcome") != "invalid_move"
    assert ("in", ("p1", "bin1")) in env.world.facts
    assert all(not (p == "obj-at" and a[0] == "p1") for p,a in env.world.facts)
    # take-out needs clear-hand; ensure it's true
    assert ("clear-hand", ("r1",)) in env.world.facts
    obs, r, done, info = TAKE(env, "p1", "bin1")
    assert info.get("outcome") != "invalid_move"
    assert ("holding", ("r1", "p1")) in env.world.facts
    assert ("in", ("p1", "bin1")) not in env.world.facts

# ---- equip / unequip -----------------------------------------------------

def test_equip_and_unequip_transitions(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=30)
    reset_with(env)
    PICK(env, "drv")
    obs, r, done, info = EQUIP(env, "drv")
    assert info.get("outcome") != "invalid_move"
    assert ("equipped", ("r1", "drv")) in env.world.facts
    assert ("holding", ("r1", "drv")) not in env.world.facts
    assert ("clear-hand", ("r1",)) in env.world.facts
    obs, r, done, info = UNEQ(env, "drv")
    assert info.get("outcome") != "invalid_move"
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
    assert info.get("outcome") != "invalid_move"
    assert ("holding", ("r1", "p_good")) in env.world.facts
    assert ("fastened", ("p_good","asm1")) not in env.world.facts
    assert ("installed", ("p_good","asm1")) not in env.world.facts
    # place it back on bench
    obs, r, done, info = BENCH(env,"p_good")
    assert info.get("outcome") != "invalid_move"
    assert ("obj-at", ("p_good","bench")) in env.world.facts
    assert ("clear-hand", ("r1",)) in env.world.facts

# ---- movement / adjacency / invariants ----------------------------------

def test_move_non_adjacent_fails(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=10)
    reset_with(env)
    obs, r, done, info = M(env, "bench", "dock")  # not directly adjacent per statics
    assert done and info.get("outcome") == "invalid_move"
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
    assert done and info.get("outcome") == "invalid_move"
    # recharge at dock
    reset_with(env, energy=2)   # enough to get bench->cell->dock
    M(env,"bench","cell"); M(env,"cell","dock")
    obs, r, done, info = RECH(env, "dock")
    assert info.get("outcome") != "invalid"
    assert env.world.get_fluent("energy", ("r1",)) >= 5

def test_time_increases(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=20)
    reset_with(env)
    t0 = env.time
    obs, r, done, info = M(env, "bench","cell")  # duration 1
    t1 = env.time
    assert t1 >= t0 + 1 - 1e-9

def test_wait_and_time_limit_enforced(asm_task_file):
    env, _ = load_task(None, str(asm_task_file), max_steps=10, time_limit=2.0)
    env.enable_durations = True
    reset_with(env)
    obs, r, done, info = env.step("(wait 2)")
    assert not done, "time == limit should not end the episode"
    obs, r, done, info = env.step("(wait 0.01)")
    assert done and info.get("outcome") == "timeout" and info.get("reason") == "time_limit_exceeded"

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

# ==============================================================================
# TASK LOADER RENAMING / VARIANT REALIZATION TESTS
# ==============================================================================

def test_termination_and_reference_plan_are_renamed_and_executable(tmp_path):
    """
    Minimal assembly task with placeholders + variants:
    - termination.when uses (and ...)
    - reference_plan uses placeholders
    We assert loader rewrites both and env reaches SUCCESS.
    """
    yaml_text = f"""
id: t_loader_asm
name: loader_renaming
meta:
  domain: assembly
  seed: 1
objects:
  r:    {{types: [robot], variants: [r1]}}
  bench: {{types: [location], variants: [bench]}}
  asm:  {{types: [assembly], variants: [asm1]}}
  p:    {{types: [part], variants: [p1]}}
  drv:  {{types: [tool], variants: [drv]}}
static_facts:
  - (tool-for drv asm)
  - (adjacent bench bench)   # degenerate but allows "move bench bench" no-op
init:
  - (at r bench)
  - (obj-at asm bench)
  - (obj-at p bench)
  - (obj-at drv bench)
  - (clear-hand r)
termination:
  - name: goal
    when: "(and (fastened p asm) (equipped r drv))"
    outcome: success
reference_plan:
  - (pick-up r drv)
  - (equip-tool r drv)
  - (align r p asm)
  - (fasten r p asm drv)
"""
    p = tmp_path / "t_loader_asm.yaml"
    p.write_text(yaml_text)
    env, meta = load_task(None, str(p), max_steps=15)

    # --- Ensure rename applied (placeholders gone)
    when = meta["termination"][0]["when"]
    assert "p " not in when and "asm " not in when and "r " not in when
    assert "(fastened p1 asm1)" in when or "(equipped r1 drv)" in when

    plan = meta["reference_plan"]
    assert any("pick-up r1 drv" in step for step in plan)

    # --- Execute plan: should hit SUCCESS at end
    obs, info = env.reset()
    for step in plan:
        obs, r, done, info = env.step(f"{step}")
        if done:
            break
    assert done and info.get("outcome") == "success"


def test_init_fluents_args_get_renamed(tmp_path):
    """
    Placeholders in init_fluents args must be rewritten to chosen variant names.
    """
    yaml_text = """
id: t_loader_fluents
name: loader_fluents
meta:
  domain: assembly
  init_fluents:
    - ["energy", ["r"], 5]
objects:
  r: {types: [robot], variants: [r1]}
  bench: {types: [location], variants: [bench]}
static_facts: []
init: ["(at r bench)"]
termination: []
"""
    p = tmp_path / "t_loader_fluents.yaml"
    p.write_text(yaml_text)
    env, _ = load_task(None, str(p))
    obs, info = env.reset()
    assert env.world.get_fluent("energy", ("r1",)) == 5.0


def test_variant_pool_collision_raises(tmp_path):
    """
    Two placeholders forced to share a single variant -> should raise ValueError.
    """
    yaml_text = """
id: t_loader_collision
name: loader_collision
meta:
  domain: assembly
objects:
  a: {types: [part], variants: [dup]}
  b: {types: [part], variants: [dup]}
static_facts: []
init: []
termination: []
"""
    p = tmp_path / "t_loader_collision.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError):
        load_task(None, str(p))


def test_head_symbols_not_renamed(tmp_path):
    """
    Verify that only object atoms are renamed, not predicate/function heads.
    Uses heads that exist in the assembly domain: and, not, >=, damaged, quality.
    """
    yaml_text = """
id: t_loader_heads
name: loader_heads
meta:
  domain: assembly
objects:
  a: {types: [part], variants: [p1]}
static_facts: []
init: []
termination:
  - name: goal
    when: "(and (not (damaged a)) (>= (quality a) 0))"
    outcome: terminal
"""
    p = tmp_path / "t_loader_heads.yaml"
    p.write_text(yaml_text)
    env, meta = load_task(None, str(p))
    when = meta["termination"][0]["when"]

    # Heads should remain intact
    for head in ("and", "not", "damaged", ">=", "quality"):
        assert head in when, f"Expected head '{head}' to remain in: {when}"

    # Placeholder should be renamed to its variant
    assert " a)" not in when
    assert " p1)" in when
