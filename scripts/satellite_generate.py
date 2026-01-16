#!/usr/bin/env python3
"""
Generate SERE Satellite tasks (domain + task YAML) faithful to IPC Satellite.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


Satellite = str
Direction = str
Instrument = str
Mode = str


def _build_domain_yaml() -> Dict:
    predicates = [
        {
            "name": "on_board",
            "args": [{"name": "i", "type": "instrument"}, {"name": "s", "type": "satellite"}],
            "static": True,
            "nl": "{i} is on board {s}",
        },
        {
            "name": "supports",
            "args": [{"name": "i", "type": "instrument"}, {"name": "m", "type": "mode"}],
            "static": True,
            "nl": "{i} supports {m}",
        },
        {
            "name": "pointing",
            "args": [{"name": "s", "type": "satellite"}, {"name": "d", "type": "direction"}],
            "nl": "{s} is pointing at {d}",
        },
        {
            "name": "power_avail",
            "args": [{"name": "s", "type": "satellite"}],
            "nl": "{s} has power available",
        },
        {
            "name": "power_on",
            "args": [{"name": "i", "type": "instrument"}],
            "nl": "{i} is powered on",
        },
        {
            "name": "calibrated",
            "args": [{"name": "i", "type": "instrument"}],
            "nl": "{i} is calibrated",
        },
        {
            "name": "have_image",
            "args": [{"name": "d", "type": "direction"}, {"name": "m", "type": "mode"}],
            "nl": "{d} has an image in {m}",
        },
        {
            "name": "calibration_target",
            "args": [{"name": "i", "type": "instrument"}, {"name": "d", "type": "direction"}],
            "static": True,
            "nl": "{i} calibrates on {d}",
        },
    ]

    turn_to = {
        "name": "turn_to",
        "params": [{"s": "satellite"}, {"d_new": "direction"}, {"d_prev": "direction"}],
        "pre": ["(pointing ?s ?d_prev)"],
        "add": ["(pointing ?s ?d_new)"],
        "del": ["(pointing ?s ?d_prev)"],
        "nl": "Turn {s} from {d_prev} to {d_new}",
        "outcomes": [
            {"name": "success", "status": "success", "p": 1.0, "add": [], "delete": []},
            {"name": "fail", "status": "fail", "p": 0.0, "add": [], "delete": []},
        ],
    }

    switch_on = {
        "name": "switch_on",
        "params": [{"i": "instrument"}, {"s": "satellite"}],
        "pre": ["(on_board ?i ?s)", "(power_avail ?s)"],
        "add": ["(power_on ?i)"],
        "del": ["(calibrated ?i)", "(power_avail ?s)"],
        "nl": "Switch on {i} on {s}",
        "outcomes": [
            {"name": "success", "status": "success", "p": 1.0, "add": [], "delete": []},
            {"name": "fail", "status": "fail", "p": 0.0, "add": [], "delete": []},
        ],
    }

    switch_off = {
        "name": "switch_off",
        "params": [{"i": "instrument"}, {"s": "satellite"}],
        "pre": ["(on_board ?i ?s)", "(power_on ?i)"],
        "add": ["(power_avail ?s)"],
        "del": ["(power_on ?i)"],
        "nl": "Switch off {i} on {s}",
        "outcomes": [
            {"name": "success", "status": "success", "p": 1.0, "add": [], "delete": []},
            {"name": "fail", "status": "fail", "p": 0.0, "add": [], "delete": []},
        ],
    }

    calibrate = {
        "name": "calibrate",
        "params": [{"s": "satellite"}, {"i": "instrument"}, {"d": "direction"}],
        "pre": [
            "(on_board ?i ?s)",
            "(calibration_target ?i ?d)",
            "(pointing ?s ?d)",
            "(power_on ?i)",
        ],
        "add": ["(calibrated ?i)"],
        "del": [],
        "nl": "Calibrate {i} on {s} at {d}",
        "outcomes": [
            {"name": "success", "status": "success", "p": 1.0, "add": [], "delete": []},
            {"name": "fail", "status": "fail", "p": 0.0, "add": [], "delete": []},
        ],
    }

    take_image = {
        "name": "take_image",
        "params": [{"s": "satellite"}, {"d": "direction"}, {"i": "instrument"}, {"m": "mode"}],
        "pre": [
            "(calibrated ?i)",
            "(on_board ?i ?s)",
            "(supports ?i ?m)",
            "(power_on ?i)",
            "(pointing ?s ?d)",
        ],
        "add": ["(have_image ?d ?m)"],
        "del": [],
        "nl": "Take {m} image of {d} using {i}",
        "outcomes": [
            {"name": "success", "status": "success", "p": 1.0, "add": [], "delete": []},
            {"name": "fail", "status": "fail", "p": 0.0, "add": [], "delete": []},
        ],
    }

    return {
        "domain": "satellite",
        "requirements": [":strips", ":typing"],
        "types": [
            {"name": "satellite"},
            {"name": "direction"},
            {"name": "instrument"},
            {"name": "mode"},
        ],
        "predicates": predicates,
        "actions": [turn_to, switch_on, switch_off, calibrate, take_image],
    }


def _assign_supports(
    instruments: Sequence[Instrument],
    modes: Sequence[Mode],
    rng: random.Random,
    density: int,
) -> Tuple[Dict[Instrument, List[Mode]], Dict[Mode, Instrument], Dict[Instrument, Mode]]:
    if not instruments or not modes:
        raise ValueError("instruments and modes must be non-empty")
    supports: Dict[Instrument, List[Mode]] = {i: [] for i in instruments}

    shuffled_insts = list(instruments)
    rng.shuffle(shuffled_insts)
    mode_to_inst: Dict[Mode, Instrument] = {}
    inst_primary_mode: Dict[Instrument, Mode] = {}

    for idx, m in enumerate(modes):
        inst = shuffled_insts[idx % len(shuffled_insts)]
        supports[inst].append(m)
        mode_to_inst[m] = inst
        if inst not in inst_primary_mode:
            inst_primary_mode[inst] = m

    for inst in instruments:
        if inst not in inst_primary_mode:
            choice = rng.choice(list(modes))
            inst_primary_mode[inst] = choice
            supports[inst].append(choice)

    for inst in instruments:
        for m in modes:
            if m in supports[inst]:
                continue
            if rng.random() < density / 100.0:
                supports[inst].append(m)

    return supports, mode_to_inst, inst_primary_mode


def _select_image_goals(
    directions: Sequence[Direction],
    modes: Sequence[Mode],
    inst_primary_mode: Dict[Instrument, Mode],
    num_images: int,
    rng: random.Random,
) -> List[Tuple[Direction, Mode]]:
    pairs = [(d, m) for d in directions for m in modes]
    if num_images > len(pairs):
        raise ValueError("num_images exceeds direction/mode pairs")

    goals: List[Tuple[Direction, Mode]] = []
    used = set()
    for mode in inst_primary_mode.values():
        if len(goals) >= num_images:
            break
        d = rng.choice(list(directions))
        pair = (d, mode)
        if pair not in used:
            goals.append(pair)
            used.add(pair)

    while len(goals) < num_images:
        pair = rng.choice(pairs)
        if pair in used:
            continue
        goals.append(pair)
        used.add(pair)
    return goals


def _build_plan(
    *,
    satellites: Sequence[Satellite],
    instruments: Sequence[Instrument],
    inst_to_sat: Dict[Instrument, Satellite],
    cal_target: Dict[Instrument, Direction],
    start_pointing: Dict[Satellite, Direction],
    images: Sequence[Tuple[Direction, Mode]],
    mode_to_inst: Dict[Mode, Instrument],
) -> List[str]:
    pointing = dict(start_pointing)
    powered: Dict[Satellite, Optional[Instrument]] = {s: None for s in satellites}
    inst_images: Dict[Instrument, List[Tuple[Direction, Mode]]] = defaultdict(list)

    for d, m in images:
        inst_images[mode_to_inst[m]].append((d, m))

    plan: List[str] = []
    for s in satellites:
        insts = [i for i in instruments if inst_to_sat[i] == s]
        for inst in insts:
            if not inst_images.get(inst):
                continue
            current = powered[s]
            if current and current != inst:
                plan.append(f"(switch_off {current} {s})")
                powered[s] = None
            if powered[s] != inst:
                plan.append(f"(switch_on {inst} {s})")
                powered[s] = inst

            cal_dir = cal_target[inst]
            if pointing[s] != cal_dir:
                plan.append(f"(turn_to {s} {cal_dir} {pointing[s]})")
                pointing[s] = cal_dir
            plan.append(f"(calibrate {s} {inst} {cal_dir})")

            for d, m in inst_images[inst]:
                if pointing[s] != d:
                    plan.append(f"(turn_to {s} {d} {pointing[s]})")
                    pointing[s] = d
                plan.append(f"(take_image {s} {d} {inst} {m})")

    return plan


def _build_task_yaml(
    *,
    task_id: str,
    name: str,
    description: str,
    satellites: Sequence[Satellite],
    directions: Sequence[Direction],
    instruments: Sequence[Instrument],
    modes: Sequence[Mode],
    start_pointing: Dict[Satellite, Direction],
    supports: Dict[Instrument, List[Mode]],
    inst_to_sat: Dict[Instrument, Satellite],
    cal_target: Dict[Instrument, Direction],
    images: Sequence[Tuple[Direction, Mode]],
    plan: List[str],
) -> Dict:
    objects: Dict[str, str] = {}
    for s in satellites:
        objects[s] = "satellite"
    for d in directions:
        objects[d] = "direction"
    for i in instruments:
        objects[i] = "instrument"
    for m in modes:
        objects[m] = "mode"

    static_facts = []
    for i in instruments:
        static_facts.append(f"(on_board {i} {inst_to_sat[i]})")
        static_facts.append(f"(calibration_target {i} {cal_target[i]})")
        for m in supports[i]:
            static_facts.append(f"(supports {i} {m})")

    init = []
    for s in satellites:
        init.append(f"(pointing {s} {start_pointing[s]})")
        init.append(f"(power_avail {s})")

    if len(images) == 1:
        d, m = images[0]
        goal = f"(have_image {d} {m})"
    else:
        goal_parts = " ".join(f"(have_image {d} {m})" for d, m in images)
        goal = f"(and {goal_parts})"

    max_steps = max(5, len(plan) + 5)

    return {
        "id": task_id,
        "name": name,
        "description": description,
        "meta": {
            "domain": "satellite",
            "enable_numeric": False,
            "enable_conditional": False,
            "enable_durations": False,
            "enable_stochastic": False,
            "max_steps": max_steps,
        },
        "objects": objects,
        "static_facts": static_facts,
        "init": init,
        "termination": [{"name": "goal", "when": goal, "outcome": "success", "reward": 1.0}],
        "reference_plan": plan,
    }


def _dump_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=False))
        f.write("\n")


def generate_task(
    *,
    task_id: str,
    name: str,
    description: str,
    num_sats: int,
    num_dirs: int,
    num_instruments: int,
    num_modes: int,
    num_images: int,
    support_density: int,
    seed: int,
    task_dir: Path,
    domain_path: Path,
) -> None:
    rng = random.Random(seed)
    satellites = [f"s{i+1}" for i in range(num_sats)]
    directions = [f"d{i+1}" for i in range(num_dirs)]
    instruments = [f"i{i+1}" for i in range(num_instruments)]
    modes = [f"m{i+1}" for i in range(num_modes)]

    if num_instruments < 1 or num_modes < 1 or num_dirs < 1 or num_sats < 1:
        raise ValueError("Need at least 1 satellite/instrument/mode/direction")
    if num_images < 1:
        raise ValueError("Need at least 1 image goal")

    inst_to_sat = {inst: satellites[idx % num_sats] for idx, inst in enumerate(instruments)}
    cal_target = {inst: rng.choice(directions) for inst in instruments}
    supports, mode_to_inst, inst_primary_mode = _assign_supports(instruments, modes, rng, support_density)
    start_pointing = {s: rng.choice(directions) for s in satellites}
    images = _select_image_goals(directions, modes, inst_primary_mode, num_images, rng)

    plan = _build_plan(
        satellites=satellites,
        instruments=instruments,
        inst_to_sat=inst_to_sat,
        cal_target=cal_target,
        start_pointing=start_pointing,
        images=images,
        mode_to_inst=mode_to_inst,
    )

    domain_yaml = _build_domain_yaml()
    task_yaml = _build_task_yaml(
        task_id=task_id,
        name=name,
        description=description,
        satellites=satellites,
        directions=directions,
        instruments=instruments,
        modes=modes,
        start_pointing=start_pointing,
        supports=supports,
        inst_to_sat=inst_to_sat,
        cal_target=cal_target,
        images=images,
        plan=plan,
    )

    _dump_yaml(domain_path, domain_yaml)
    _dump_yaml(task_dir / f"{task_id}.yaml", task_yaml)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a SERE Satellite task.")
    ap.add_argument("--task-id", required=True, help="Task id, e.g. t01_satellite_easy_a")
    ap.add_argument("--name", default=None, help="Human-readable task name")
    ap.add_argument("--rationale", default=None, help="Short rationale appended to the description")
    ap.add_argument("--satellites", type=int, required=True, help="Number of satellites")
    ap.add_argument("--directions", type=int, required=True, help="Number of directions")
    ap.add_argument("--instruments", type=int, required=True, help="Number of instruments")
    ap.add_argument("--modes", type=int, required=True, help="Number of modes")
    ap.add_argument("--images", type=int, required=True, help="Number of image goals")
    ap.add_argument("--support-density", type=int, default=50, help="Extra instrument/mode coverage percent")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument(
        "--task-dir",
        default="src/sere/assets/tasks/satellite",
        help="Output directory for task YAML",
    )
    ap.add_argument(
        "--domain-path",
        default="src/sere/assets/domain/satellite.yaml",
        help="Output path for the domain YAML",
    )
    args = ap.parse_args()

    name = args.name or f"Satellite {args.task_id}"
    desc = (
        "Turn, calibrate, and capture images. "
        f"Settings: satellites={args.satellites}, directions={args.directions}, "
        f"instruments={args.instruments}, modes={args.modes}, images={args.images}."
    )
    if args.rationale:
        desc = desc.rstrip(".") + f". Rationale: {args.rationale}"

    generate_task(
        task_id=args.task_id,
        name=name,
        description=desc,
        num_sats=args.satellites,
        num_dirs=args.directions,
        num_instruments=args.instruments,
        num_modes=args.modes,
        num_images=args.images,
        support_density=args.support_density,
        seed=args.seed,
        task_dir=Path(args.task_dir),
        domain_path=Path(args.domain_path),
    )

    print(
        f"Generated {args.task_id} (sats={args.satellites}, directions={args.directions}, "
        f"instruments={args.instruments}, modes={args.modes}, images={args.images})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
