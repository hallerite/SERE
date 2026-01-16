#!/usr/bin/env python3
"""
Generate SERE Rovers tasks (IPC-style rovers with communication + visibility).
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Waypoint = str


def _adjacency(edges: Sequence[Tuple[Waypoint, Waypoint]]) -> Dict[Waypoint, List[Waypoint]]:
    adj: Dict[Waypoint, List[Waypoint]] = {}
    for a, b in edges:
        adj.setdefault(a, [])
        adj.setdefault(b, [])
        if b not in adj[a]:
            adj[a].append(b)
        if a not in adj[b]:
            adj[b].append(a)
    return adj


def _shortest_path(adj: Dict[Waypoint, List[Waypoint]], start: Waypoint, goal: Waypoint) -> List[Waypoint]:
    if start == goal:
        return [start]
    q: deque[Waypoint] = deque([start])
    prev: Dict[Waypoint, Waypoint] = {}
    seen = {start}
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            prev[nxt] = cur
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in seen:
        raise ValueError(f"No path from {start} to {goal}")
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _moves_for_path(path: Sequence[Waypoint], rover: str) -> List[str]:
    moves: List[str] = []
    for a, b in zip(path, path[1:]):
        moves.append(f"(navigate {rover} {a} {b})")
    return moves


def _settings_text(spec: dict) -> str:
    return (
        f"Settings: waypoints={len(spec['waypoints'])}, samples={len(spec['samples'])}, "
        f"images={len(spec['images'])}, cameras={len(spec['cameras'])}."
    )


def _build_plan(spec: dict) -> List[str]:
    rover = spec["rovers"][0]
    store = spec["stores"][0]
    lander = spec["lander"]
    base = spec["lander_at"]

    adj = _adjacency(spec["can_traverse"])
    plan: List[str] = []
    current = base

    for sample in spec["samples"]:
        dest = sample["waypoint"]
        path = _shortest_path(adj, current, dest)
        plan.extend(_moves_for_path(path, rover))
        current = dest
        action = "sample_soil" if sample["type"] == "soil" else "sample_rock"
        plan.append(f"({action} {rover} {store} {dest})")
        plan.append(f"(communicate_{sample['type']}_data {rover} {lander} {dest} {current} {base})")
        plan.append(f"(drop {rover} {store})")

    for img in spec["images"]:
        cam = img["camera"]
        mode = img["mode"]
        obj = img["objective"]

        cal_obj = spec["calibration_targets"][cam]
        cal_wp = spec["objective_visible_from"][cal_obj][0]
        if current != cal_wp:
            path = _shortest_path(adj, current, cal_wp)
            plan.extend(_moves_for_path(path, rover))
            current = cal_wp
        plan.append(f"(calibrate {rover} {cam} {cal_obj} {cal_wp})")

        img_wp = spec["objective_visible_from"][obj][0]
        if current != img_wp:
            path = _shortest_path(adj, current, img_wp)
            plan.extend(_moves_for_path(path, rover))
            current = img_wp
        plan.append(f"(take_image {rover} {img_wp} {obj} {cam} {mode})")
        plan.append(f"(communicate_image_data {rover} {lander} {obj} {mode} {current} {base})")

    return plan


def _goal_expr(samples: Sequence[dict], images: Sequence[dict]) -> str:
    goals = []
    for sample in samples:
        if sample["type"] == "soil":
            goals.append(f"(communicated_soil_data {sample['waypoint']})")
        else:
            goals.append(f"(communicated_rock_data {sample['waypoint']})")
    goals.extend(f"(communicated_image_data {img['objective']} {img['mode']})" for img in images)
    if not goals:
        return "(and)"
    if len(goals) == 1:
        return goals[0]
    return f"(and {' '.join(goals)})"


def _task_yaml(spec: dict) -> dict:
    rover = spec["rovers"][0]
    store = spec["stores"][0]
    lander = spec["lander"]
    base = spec["lander_at"]

    objects: Dict[str, str] = {}
    for r in spec["rovers"]:
        objects[r] = "rover"
    for w in spec["waypoints"]:
        objects[w] = "waypoint"
    for s in spec["stores"]:
        objects[s] = "store"
    for c in spec["cameras"]:
        objects[c] = "camera"
    for m in spec["modes"]:
        objects[m] = "mode"
    for o in spec["objectives"]:
        objects[o] = "objective"
    objects[lander] = "lander"

    static_facts: List[str] = []
    static_facts.append(f"(at_lander {lander} {base})")
    for r in spec["rovers"]:
        static_facts.append(f"(equipped_for_soil_analysis {r})")
        static_facts.append(f"(equipped_for_rock_analysis {r})")
        static_facts.append(f"(equipped_for_imaging {r})")
    for a, b in spec["can_traverse"]:
        static_facts.append(f"(can_traverse {rover} {a} {b})")
        static_facts.append(f"(can_traverse {rover} {b} {a})")

    visible_edges = set()
    for a, b in spec["can_traverse"]:
        visible_edges.add((a, b))
        visible_edges.add((b, a))
    for w in spec["waypoints"]:
        visible_edges.add((base, w))
        visible_edges.add((w, base))

    for a, b in sorted(visible_edges):
        static_facts.append(f"(visible {a} {b})")
    for s in spec["stores"]:
        static_facts.append(f"(store_of {s} {rover})")
    for cam in spec["cameras"]:
        static_facts.append(f"(on_board {cam} {rover})")
        for mode in spec["camera_supports"][cam]:
            static_facts.append(f"(supports {cam} {mode})")
        static_facts.append(f"(calibration_target {cam} {spec['calibration_targets'][cam]})")
    for obj, wps in spec["objective_visible_from"].items():
        for wp in wps:
            static_facts.append(f"(visible_from {obj} {wp})")

    init = [f"(at {rover} {base})", f"(empty {store})", f"(available {rover})", f"(channel_free {lander})"]
    for sample in spec["samples"]:
        if sample["type"] == "soil":
            init.append(f"(at_soil_sample {sample['waypoint']})")
        else:
            init.append(f"(at_rock_sample {sample['waypoint']})")

    plan = _build_plan(spec)
    description = f"{spec['description']} {_settings_text(spec)} Rationale: {spec['rationale']}"

    return {
        "id": spec["id"],
        "name": spec["name"],
        "description": description,
        "meta": {
            "domain": "rovers",
            "enable_numeric": False,
            "enable_conditional": False,
            "enable_durations": False,
            "enable_stochastic": False,
            "max_steps": len(plan) + 5,
        },
        "objects": objects,
        "static_facts": static_facts,
        "init": init,
        "termination": [
            {
                "name": "goal",
                "when": _goal_expr(spec["samples"], spec["images"]),
                "outcome": "success",
                "reward": 1.0,
            }
        ],
        "reference_plan": plan,
    }


def _specs() -> List[dict]:
    return [
        {
            "id": "t01_rovers_easy_a",
            "name": "Rovers Easy A",
            "description": "Collect a soil sample and transmit one image on a short line.",
            "rationale": "Single rover, one camera, and one objective with local calibration.",
            "rovers": ["r1"],
            "stores": ["s1"],
            "lander": "l1",
            "lander_at": "w1",
            "waypoints": ["w1", "w2", "w3"],
            "can_traverse": [("w1", "w2"), ("w2", "w3")],
            "cameras": ["c1"],
            "modes": ["m1"],
            "camera_supports": {"c1": ["m1"]},
            "calibration_targets": {"c1": "o1"},
            "objectives": ["o1"],
            "objective_visible_from": {"o1": ["w3"]},
            "samples": [{"type": "soil", "waypoint": "w2"}],
            "images": [{"camera": "c1", "mode": "m1", "objective": "o1"}],
        },
        {
            "id": "t02_rovers_easy_b",
            "name": "Rovers Easy B",
            "description": "Collect a rock sample and transmit one image with a calibration detour.",
            "rationale": "Separates calibration and imaging waypoints while keeping a small map.",
            "rovers": ["r1"],
            "stores": ["s1"],
            "lander": "l1",
            "lander_at": "w1",
            "waypoints": ["w1", "w2", "w3"],
            "can_traverse": [("w1", "w2"), ("w2", "w3")],
            "cameras": ["c1"],
            "modes": ["m1"],
            "camera_supports": {"c1": ["m1"]},
            "calibration_targets": {"c1": "o1"},
            "objectives": ["o1", "o2"],
            "objective_visible_from": {"o1": ["w2"], "o2": ["w3"]},
            "samples": [{"type": "rock", "waypoint": "w2"}],
            "images": [{"camera": "c1", "mode": "m1", "objective": "o2"}],
        },
        {
            "id": "t03_rovers_medium_a",
            "name": "Rovers Medium A",
            "description": "Two samples and two images along a longer route.",
            "rationale": "Repeated calibration and communication across four waypoints.",
            "rovers": ["r1"],
            "stores": ["s1"],
            "lander": "l1",
            "lander_at": "w1",
            "waypoints": ["w1", "w2", "w3", "w4"],
            "can_traverse": [("w1", "w2"), ("w2", "w3"), ("w3", "w4")],
            "cameras": ["c1"],
            "modes": ["m1"],
            "camera_supports": {"c1": ["m1"]},
            "calibration_targets": {"c1": "o1"},
            "objectives": ["o1", "o2", "o3"],
            "objective_visible_from": {"o1": ["w2"], "o2": ["w4"], "o3": ["w3"]},
            "samples": [
                {"type": "soil", "waypoint": "w2"},
                {"type": "rock", "waypoint": "w4"},
            ],
            "images": [
                {"camera": "c1", "mode": "m1", "objective": "o2"},
                {"camera": "c1", "mode": "m1", "objective": "o3"},
            ],
        },
        {
            "id": "t04_rovers_medium_b",
            "name": "Rovers Medium B",
            "description": "Two samples with two imaging modes on a branching map.",
            "rationale": "Multiple modes require more calibrations across branches.",
            "rovers": ["r1"],
            "stores": ["s1"],
            "lander": "l1",
            "lander_at": "w1",
            "waypoints": ["w1", "w2", "w3", "w4", "w5"],
            "can_traverse": [("w1", "w2"), ("w1", "w3"), ("w3", "w4"), ("w3", "w5")],
            "cameras": ["c1"],
            "modes": ["m1", "m2"],
            "camera_supports": {"c1": ["m1", "m2"]},
            "calibration_targets": {"c1": "o1"},
            "objectives": ["o1", "o2", "o3"],
            "objective_visible_from": {"o1": ["w3"], "o2": ["w4"], "o3": ["w5"]},
            "samples": [
                {"type": "soil", "waypoint": "w2"},
                {"type": "rock", "waypoint": "w4"},
            ],
            "images": [
                {"camera": "c1", "mode": "m1", "objective": "o2"},
                {"camera": "c1", "mode": "m2", "objective": "o3"},
            ],
        },
        {
            "id": "t05_rovers_hard_a",
            "name": "Rovers Hard A",
            "description": "Three samples and three images with two cameras.",
            "rationale": "Longer traversal and mixed camera usage with repeated calibration.",
            "rovers": ["r1"],
            "stores": ["s1"],
            "lander": "l1",
            "lander_at": "w1",
            "waypoints": ["w1", "w2", "w3", "w4", "w5", "w6"],
            "can_traverse": [("w1", "w2"), ("w2", "w3"), ("w3", "w4"), ("w4", "w5"), ("w5", "w6")],
            "cameras": ["c1", "c2"],
            "modes": ["m1", "m2"],
            "camera_supports": {"c1": ["m1"], "c2": ["m2"]},
            "calibration_targets": {"c1": "o1", "c2": "o2"},
            "objectives": ["o1", "o2", "o3", "o4"],
            "objective_visible_from": {"o1": ["w2"], "o2": ["w4"], "o3": ["w5"], "o4": ["w6"]},
            "samples": [
                {"type": "soil", "waypoint": "w2"},
                {"type": "rock", "waypoint": "w3"},
                {"type": "soil", "waypoint": "w5"},
            ],
            "images": [
                {"camera": "c1", "mode": "m1", "objective": "o3"},
                {"camera": "c2", "mode": "m2", "objective": "o4"},
                {"camera": "c1", "mode": "m1", "objective": "o2"},
            ],
        },
        {
            "id": "t06_rovers_hard_b",
            "name": "Rovers Hard B",
            "description": "Three samples and four images across two cameras and three modes.",
            "rationale": "Largest map with repeated calibrations and mixed mode goals.",
            "rovers": ["r1"],
            "stores": ["s1"],
            "lander": "l1",
            "lander_at": "w1",
            "waypoints": ["w1", "w2", "w3", "w4", "w5", "w6", "w7"],
            "can_traverse": [
                ("w1", "w2"),
                ("w2", "w3"),
                ("w3", "w4"),
                ("w3", "w5"),
                ("w5", "w6"),
                ("w6", "w7"),
            ],
            "cameras": ["c1", "c2"],
            "modes": ["m1", "m2", "m3"],
            "camera_supports": {"c1": ["m1", "m2"], "c2": ["m2", "m3"]},
            "calibration_targets": {"c1": "o1", "c2": "o2"},
            "objectives": ["o1", "o2", "o3", "o4", "o5"],
            "objective_visible_from": {"o1": ["w2"], "o2": ["w3"], "o3": ["w4"], "o4": ["w6"], "o5": ["w7"]},
            "samples": [
                {"type": "soil", "waypoint": "w2"},
                {"type": "rock", "waypoint": "w4"},
                {"type": "soil", "waypoint": "w7"},
            ],
            "images": [
                {"camera": "c1", "mode": "m1", "objective": "o3"},
                {"camera": "c2", "mode": "m2", "objective": "o4"},
                {"camera": "c2", "mode": "m3", "objective": "o5"},
                {"camera": "c1", "mode": "m2", "objective": "o2"},
            ],
        },
    ]


def _write_task(out_dir: Path, spec: dict) -> Path:
    task = _task_yaml(spec)
    path = out_dir / f"{spec['id']}.yaml"
    path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IPC-style Rovers tasks.")
    parser.add_argument(
        "--out-dir",
        default="src/sere/assets/tasks/rovers",
        help="Output directory for tasks.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _specs()
    for spec in specs:
        _write_task(out_dir, spec)

    print(f"Wrote {len(specs)} tasks to {out_dir}")


if __name__ == "__main__":
    main()
