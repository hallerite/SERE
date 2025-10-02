# tests/test_affordance_generation.py

from sere.core.pddl_env.prompt_formatter import PromptFormatter, PromptFormatterConfig, VisibilityScope
from sere.core.pddl_env.run_mode import RunMode
from sere.pddl.domain_spec import DomainSpec, ActionSpec, PredicateSpec
from sere.core.world_state import WorldState


def _mini_domain_for_filters() -> DomainSpec:
    # Predicates used in preconditions
    preds = {
        "at":         PredicateSpec(name="at", args=[("r", "robot"), ("l", "loc")], nl=["{r} is at {l}"]),
        "obj-at":     PredicateSpec(name="obj-at", args=[("o", "object"), ("l", "loc")], nl=["{o} is at {l}"]),
        "adjacent":   PredicateSpec(name="adjacent", args=[("a", "loc"), ("b", "loc")], nl=["{a} is adjacent to {b}"]),
        "clear-hand": PredicateSpec(name="clear-hand", args=[("r", "robot")], nl=["{r} has a free hand"]),
        "holding":    PredicateSpec(name="holding", args=[("r", "robot"), ("o", "object")], nl=["{r} holds {o}"]),
    }

    # Actions:
    # move(r, s, d): only forward direction from the true source (pre at(r,s))
    move = ActionSpec(
        name="move",
        params=[("r","robot"),("s","loc"),("d","loc")],
        pre=["(at ?r ?s)"],
        add=["(at ?r ?d)"],
        delete=["(at ?r ?s)"],
        nl=["move {r} from {s} to {d}"],
    )

    # pick-up(r, o): same room + clear hand (robot must be at kitchen for this toy test)
    pickup = ActionSpec(
        name="pick-up",
        params=[("r", "robot"), ("o", "object")],
        pre=["(at ?r kitchen)", "(obj-at ?o kitchen)", "(clear-hand ?r)"],
        add=["(holding ?r ?o)"],
        delete=["(obj-at ?o kitchen)", "(clear-hand ?r)"],
        nl=["pick up {o}"],
    )

    return DomainSpec(
        name="filters-mini",
        types={},  # WorldState carries types for these tests
        predicates=preds,
        actions={"move": move, "pick-up": pickup},
        fluents={},
    )


def _mini_world_for_filters(domain: DomainSpec) -> WorldState:
    w = WorldState(domain=domain, objects={}, facts=set(), fluents={})
    # objects & their types
    w.objects["r1"] = {"robot"}
    w.objects["kitchen"] = {"loc"}
    w.objects["pantry"] = {"loc"}
    w.objects["leaf"] = {"object"}

    # facts: robot in kitchen, leaf in pantry, rooms adjacent, free hand
    w.facts |= {
        ("at", ("r1", "kitchen")),
        ("obj-at", ("leaf", "pantry")),
        ("adjacent", ("kitchen", "pantry")),
        ("adjacent", ("pantry", "kitchen")),
        ("clear-hand", ("r1",)),
    }
    return w


def _generate_and_render(formatter: PromptFormatter, world: WorldState, static_facts=set()) -> str:
    # Generate affordances first (scoped inside the formatter)
    affs = formatter.generate_affordances(world, static_facts=static_facts, enable_numeric=True)
    # Then render (also scoped) so what you see is what the agent sees
    return formatter.format_obs(
        world=world,
        steps=0,
        max_steps=5,
        time_val=0.0,
        durations_on=False,
        messages=[],
        affordances=affs,
        time_limit=None,
        termination_rules=None,
        run_mode=RunMode.INTERACTIVE
    )


# --- Visibility + affordances ---------------------------------------------------

def test_affordances_visibility_all():
    domain = _mini_domain_for_filters()
    world = _mini_world_for_filters(domain)
    fmt = PromptFormatter(domain, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=False))

    obs = _generate_and_render(fmt, world)

    # With robot at kitchen and leaf at pantry:
    # - Valid move to pantry should appear (pantry is visible under ALL)
    # - Reverse move is absent (pre fails)
    # - pick-up is absent (leaf not in kitchen)
    assert "(move r1 kitchen pantry)" in obs
    assert "(move r1 pantry kitchen)" not in obs
    assert "(pick-up r1 leaf)" not in obs


def test_affordances_hidden_when_flag_off():
    domain = _mini_domain_for_filters()
    world = _mini_world_for_filters(domain)
    fmt = PromptFormatter(domain, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=False, display_nl=False))

    obs = _generate_and_render(fmt, world)

    # No affordances at all
    assert "Valid moves:" not in obs
    assert "(move r1 kitchen pantry)" not in obs
    assert "(pick-up r1 leaf)" not in obs


def test_affordances_visibility_room_scoped_generation():
    domain = _mini_domain_for_filters()
    world = _mini_world_for_filters(domain)

    # ROOM scope → object pool is sliced to the current room (kitchen), so 'pantry' is not enumerated.
    fmt = PromptFormatter(domain, PromptFormatterConfig(
        visibility=VisibilityScope.ROOM, show_affordances=True, display_nl=False))

    obs = _generate_and_render(fmt, world)

    # Expect only kitchen-local grounding; pantry isn't visible so isn't enumerated.
    assert "(move r1 kitchen kitchen)" in obs
    assert "(move r1 kitchen pantry)" not in obs
    assert "(move r1 pantry kitchen)" not in obs
    assert "(pick-up r1 leaf)" not in obs


def test_affordances_action_without_preconditions_lists_under_all():
    d = _mini_domain_for_filters()
    # Add a no-precondition action
    d.actions["noop"] = ActionSpec(
        name="noop",
        params=[],
        pre=[],
        add=[],
        delete=[],
        nl=["do nothing"],
    )
    w = _mini_world_for_filters(d)

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=False))
    obs = _generate_and_render(fmt, w)

    assert "Valid moves:" in obs
    assert "(noop)" in obs


def test_affordances_numeric_param_templated_placeholder_and_non_numeric_check():
    d = _mini_domain_for_filters()
    # add a simple predicate and an action with a numeric parameter
    d.predicates["powered"] = PredicateSpec(name="powered", args=[("a", "object")], nl=["{a} is powered"])
    d.actions["heat"] = ActionSpec(
        name="heat",
        params=[("r", "robot"), ("a", "object"), ("n", "number")],
        pre=["(powered ?a)"],  # non-numeric pre; numeric param present → show <n>
        add=[],
        delete=[],
        nl=["heat {a} for {n} ticks"],
    )

    w = _mini_world_for_filters(d)
    w.objects["heater"] = {"object"}
    w.facts.add(("powered", ("heater",)))

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=False))
    obs = _generate_and_render(fmt, w)

    assert "(heat r1 heater <n>)" in obs


def test_affordances_numeric_precondition_skipped_for_template():
    d = _mini_domain_for_filters()
    # action whose ONLY precondition is numeric; since it has a number param, numeric guard is skipped
    d.actions["dose"] = ActionSpec(
        name="dose",
        params=[("r", "robot"), ("n", "number")],
        pre=["(> (energy ?r) ?n)"],
        add=[],
        delete=[],
        nl=["dose {r} by {n}"],
    )
    w = _mini_world_for_filters(d)

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=False))
    affs = fmt.generate_affordances(w, static_facts=set(), enable_numeric=True)

    assert "(dose r1 <n>)" in affs


def test_room_scope_with_containment_chain_affords_object_in_room():
    d = _mini_domain_for_filters()
    # add an 'inspect' action touching an object arg
    d.actions["inspect"] = ActionSpec(
        name="inspect",
        params=[("r", "robot"), ("o", "object")],
        pre=["(at ?r kitchen)"],
        add=[],
        delete=[],
        nl=["inspect {o}"],
    )
    w = _mini_world_for_filters(d)
    # place leaf inside cup, and cup in the kitchen (so leaf resolves to kitchen)
    w.objects["cup"] = {"object"}
    w.facts.discard(("obj-at", ("leaf", "pantry")))
    w.facts.add(("in", ("leaf", "cup")))
    w.facts.add(("obj-at", ("cup", "kitchen")))

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ROOM, show_affordances=True, display_nl=False))
    obs = _generate_and_render(fmt, w)

    assert "(inspect r1 leaf)" in obs


def test_affordances_use_static_facts_in_preconditions():
    d = _mini_domain_for_filters()
    # move2 requires adjacency; we remove adjacency from dynamic facts and supply as static
    d.actions["move2"] = ActionSpec(
        name="move2",
        params=[("r", "robot"), ("s", "loc"), ("d", "loc")],
        pre=["(at ?r ?s)", "(adjacent ?s ?d)"],
        add=["(at ?r ?d)"],
        delete=["(at ?r ?s)"],
        nl=["move2 {r} from {s} to {d}"],
    )

    w = _mini_world_for_filters(d)
    w.facts.discard(("adjacent", ("kitchen", "pantry")))
    w.facts.discard(("adjacent", ("pantry", "kitchen")))

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=False))
    affs = fmt.generate_affordances(
        w,
        static_facts={("adjacent", ("kitchen", "pantry"))},
        enable_numeric=True,
    )
    assert "(move2 r1 kitchen pantry)" in affs


def test_type_pool_empty_yields_no_affordance():
    d = _mini_domain_for_filters()
    d.actions["open"] = ActionSpec(
        name="open",
        params=[("c", "container")],
        pre=[],
        add=[],
        delete=[],
        nl=["open {c}"],
    )
    w = _mini_world_for_filters(d)

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=False))
    affs = fmt.generate_affordances(w, static_facts=set(), enable_numeric=True)

    assert "(open" not in " ".join(affs)


def test_multi_typed_object_appears_in_multiple_type_pools():
    d = _mini_domain_for_filters()
    d.actions["use-appliance"] = ActionSpec(
        name="use-appliance",
        params=[("r", "robot"), ("a", "appliance")],
        pre=["(at ?r kitchen)"],
        add=[],
        delete=[],
        nl=["use {a}"],
    )
    d.actions["store-in-container"] = ActionSpec(
        name="store-in-container",
        params=[("r", "robot"), ("c", "container")],
        pre=["(at ?r kitchen)"],
        add=[],
        delete=[],
        nl=["store in {c}"],
    )

    w = _mini_world_for_filters(d)
    w.objects["heater"] = {"appliance", "container"}

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=False))
    affs = fmt.generate_affordances(w, static_facts=set(), enable_numeric=True)

    assert "(use-appliance r1 heater)" in affs
    assert "(store-in-container r1 heater)" in affs


def test_affordances_render_with_nl_when_enabled():
    d = _mini_domain_for_filters()
    w = _mini_world_for_filters(d)

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=True))
    obs = _generate_and_render(fmt, w)

    assert "Valid moves:" in obs
    assert "move r1 from kitchen to pantry" in obs  # NL description appended


def test_room_scope_generation_blocks_actions_not_touching_current_room():
    d = _mini_domain_for_filters()
    # action referencing a location param
    d.actions["announce"] = ActionSpec(
        name="announce",
        params=[("l", "loc")],
        pre=[],
        add=[],
        delete=[],
        nl=["announce {l}"],
    )
    w = _mini_world_for_filters(d)

    # Under ALL, both locations are enumerated
    fmt_all = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=False))
    affs_all = fmt_all.generate_affordances(w, static_facts=set(), enable_numeric=True)
    assert "(announce kitchen)" in affs_all
    assert "(announce pantry)" in affs_all

    # Under ROOM, only 'kitchen' is visible → pantry is not enumerated at all
    fmt_room = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ROOM, show_affordances=True, display_nl=False))
    affs_room = fmt_room.generate_affordances(w, static_facts=set(), enable_numeric=True)
    assert "(announce kitchen)" in affs_room
    assert "(announce pantry)" not in affs_room


# --- System prompt visibility ---------------------------------------------------

def test_system_prompt_visibility_all_shows_all_objects_and_statics():
    d = _mini_domain_for_filters()
    w = _mini_world_for_filters(d)

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=False, display_nl=False,
        show_objects_in_sysprompt=True, show_briefing=True
    ))
    
    sys = fmt.build_system_prompt(
        world=w,
        static_facts={("adjacent", ("kitchen", "pantry"))},
        run_mode=RunMode.INTERACTIVE,
    )

    # Objects should list kitchen, pantry, r1, leaf
    assert "Objects (name - type):" in sys
    assert "kitchen: {loc}" in sys
    assert "pantry: {loc}" in sys
    assert "r1: {robot}" in sys
    assert "leaf: {object}" in sys

    # Statics should include adjacency in ALL visibility
    assert "Statics (do not change):" in sys
    assert "- (adjacent kitchen pantry)" in sys


def test_system_prompt_visibility_room_hides_nonlocal_objects_and_statics():
    d = _mini_domain_for_filters()
    w = _mini_world_for_filters(d)

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ROOM, show_affordances=False, display_nl=False,
        show_objects_in_sysprompt=True, show_briefing=True
    ))
    
    sys = fmt.build_system_prompt(
        world=w,
        static_facts={("adjacent", ("kitchen", "pantry"))},
        run_mode=RunMode.INTERACTIVE,
    )

    # Objects should only list r1 and kitchen (leaf in pantry is not visible; pantry itself not shown)
    assert "Objects (name - type):" in sys
    assert "r1: {robot}" in sys
    assert "kitchen: {loc}" in sys
    assert "pantry: {loc}" not in sys
    assert "leaf: {object}" not in sys

    # Adjacency statics are hidden under ROOM visibility
    assert "Statics (do not change):" in sys
    assert "- (adjacent kitchen pantry)" not in sys

def test_state_room_visibility_hides_nonlocal_facts():
    d = _mini_domain_for_filters()
    w = _mini_world_for_filters(d)

    # ROOM scope should hide non-local facts (leaf is in pantry)
    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ROOM, show_affordances=False, display_nl=False
    ))
    obs = _generate_and_render(fmt, w)

    assert "State:" in obs
    # robot location is always shown
    assert "- (at r1 kitchen)" in obs
    # non-local fact should be hidden
    assert "- (obj-at leaf pantry)" not in obs


def test_fluent_visibility_room_shows_robot_energy_hides_nonlocal_object_and_shows_local_via_containment():
    d = _mini_domain_for_filters()
    w = _mini_world_for_filters(d)

    # add local container + contained object in the kitchen
    w.objects["mug"] = {"object"}
    w.objects["water"] = {"object"}
    w.facts.add(("obj-at", ("mug", "kitchen")))
    w.facts.add(("in", ("water", "mug")))  # containment chain → water resolves to kitchen

    # fluents: robot energy (always visible), local temp (water), non-local temp (leaf in pantry)
    w.fluents[("energy", ("r1",))] = 3.0
    w.fluents[("temp", ("water",))] = 60.0
    w.fluents[("temp", ("leaf",))] = 42.0

    fmt = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ROOM, show_affordances=False, display_nl=False
    ))
    obs = _generate_and_render(fmt, w)

    # header shows energy even in ROOM scope
    assert "Energy: r1" in obs

    # Fluents block should include local water temp but hide leaf temp (non-local)
    assert "Fluents:" in obs
    assert "- (= (temp water) 60" in obs
    assert "- (= (temp leaf) 42" not in obs


def test_room_scope_with_static_adjacency_does_not_leak_nonlocal_affordance():
    d = _mini_domain_for_filters()
    # move2 requires adjacency; we'll remove dynamic adjacency and provide static
    d.actions["move2"] = ActionSpec(
        name="move2",
        params=[("r", "robot"), ("s", "loc"), ("d", "loc")],
        pre=["(at ?r ?s)", "(adjacent ?s ?d)"],
        add=["(at ?r ?d)"],
        delete=["(at ?r ?s)"],
        nl=["move2 {r} from {s} to {d}"],
    )

    w = _mini_world_for_filters(d)
    # strip dynamic adjacency so only the static fact exists
    w.facts.discard(("adjacent", ("kitchen", "pantry")))
    w.facts.discard(("adjacent", ("pantry", "kitchen")))

    # Under ALL, the static should enable the affordance
    fmt_all = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ALL, show_affordances=True, display_nl=False
    ))
    affs_all = fmt_all.generate_affordances(
        w, static_facts={("adjacent", ("kitchen", "pantry"))}, enable_numeric=True
    )
    assert "(move2 r1 kitchen pantry)" in affs_all

    # Under ROOM, pantry is out of scope → object pool sliced → affordance must not be enumerated
    fmt_room = PromptFormatter(d, PromptFormatterConfig(
        visibility=VisibilityScope.ROOM, show_affordances=True, display_nl=False
    ))
    affs_room = fmt_room.generate_affordances(
        w, static_facts={("adjacent", ("kitchen", "pantry"))}, enable_numeric=True
    )
    assert "(move2 r1 kitchen pantry)" not in affs_room
