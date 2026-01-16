#!/usr/bin/env python3
"""
Generate SERE Floortile tasks (IPC-style checkerboard painting).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence


def _tile(row: int, col: int) -> str:
    return f"tile_{row}-{col}"


def _color_for(row: int, col: int) -> str:
    return "white" if (row + col) % 2 == 0 else "black"


def _build_plan(
    num_rows: int,
    col_start: int,
    col_end: int,
    robot: str,
    start_color: str,
) -> List[str]:
    plan: List[str] = []
    current_col = col_start
    current_color = start_color

    def move_horizontal(target_col: int) -> None:
        nonlocal current_col
        while current_col < target_col:
            plan.append(f"(right {robot} {_tile(0, current_col)} {_tile(0, current_col + 1)})")
            current_col += 1
        while current_col > target_col:
            plan.append(f"(left {robot} {_tile(0, current_col)} {_tile(0, current_col - 1)})")
            current_col -= 1

    for col in range(col_start, col_end + 1):
        move_horizontal(col)

        # Move up to row num_rows - 1.
        for r in range(0, max(0, num_rows - 1)):
            plan.append(f"(up {robot} {_tile(r, col)} {_tile(r + 1, col)})")

        # Paint top row (num_rows) from row num_rows - 1.
        top_row = num_rows
        desired = _color_for(top_row, col)
        if desired != current_color:
            plan.append(f"(change-color {robot} {current_color} {desired})")
            current_color = desired
        plan.append(f"(paint-up {robot} {_tile(top_row, col)} {_tile(max(0, num_rows - 1), col)} {desired})")

        # Descend, painting each row after moving down.
        for r in range(num_rows - 1, 0, -1):
            plan.append(f"(down {robot} {_tile(r, col)} {_tile(r - 1, col)})")
            desired = _color_for(r, col)
            if desired != current_color:
                plan.append(f"(change-color {robot} {current_color} {desired})")
                current_color = desired
            plan.append(f"(paint-up {robot} {_tile(r, col)} {_tile(r - 1, col)} {desired})")

    return plan


def _build_task(spec: dict) -> dict:
    num_rows = spec["rows"]
    num_cols = spec["cols"]
    num_robots = spec["robots"]

    robots = [f"r{i+1}" for i in range(num_robots)]
    colors = ["white", "black"]

    if num_robots > num_cols:
        raise ValueError("num_robots must be <= num_cols")

    base = num_cols // num_robots
    extra = num_cols % num_robots
    segments = []
    col = 1
    for i in range(num_robots):
        width = base + (1 if i < extra else 0)
        start = col
        end = col + width - 1
        segments.append((start, end))
        col = end + 1

    objects: Dict[str, str] = {}
    for r in robots:
        objects[r] = "robot"
    for row in range(num_rows + 1):
        for col in range(1, num_cols + 1):
            objects[_tile(row, col)] = "tile"
    for c in colors:
        objects[c] = "color"

    static_facts: List[str] = []
    for row in range(num_rows):
        for col in range(1, num_cols + 1):
            static_facts.append(f"(up {_tile(row + 1, col)} {_tile(row, col)})")
            static_facts.append(f"(down {_tile(row, col)} {_tile(row + 1, col)})")
    for row in range(num_rows + 1):
        for col in range(1, num_cols):
            static_facts.append(f"(right {_tile(row, col + 1)} {_tile(row, col)})")
            static_facts.append(f"(left {_tile(row, col)} {_tile(row, col + 1)})")
    for c in colors:
        static_facts.append(f"(available-color {c})")

    init: List[str] = []
    robot_positions = {}
    for r, (start, _end) in zip(robots, segments):
        tile = _tile(0, start)
        robot_positions[r] = tile
        init.append(f"(robot-at {r} {tile})")

    for idx, r in enumerate(robots):
        color = "white" if idx % 2 == 0 else "black"
        init.append(f"(robot-has {r} {color})")

    occupied = set(robot_positions.values())
    for row in range(num_rows + 1):
        for col in range(1, num_cols + 1):
            tile = _tile(row, col)
            if tile in occupied:
                continue
            init.append(f"(clear {tile})")

    goal_parts = []
    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            goal_parts.append(f"(painted {_tile(row, col)} {_color_for(row, col)})")

    if len(goal_parts) == 1:
        goal_expr = goal_parts[0]
    else:
        goal_expr = f"(and {' '.join(goal_parts)})"

    plan: List[str] = []
    for idx, (r, (start, end)) in enumerate(zip(robots, segments)):
        start_color = "white" if idx % 2 == 0 else "black"
        plan.extend(_build_plan(num_rows, start, end, r, start_color))

    return {
        "id": spec["id"],
        "name": spec["name"],
        "description": f"{spec['description']} Settings: rows={num_rows}, cols={num_cols}, robots={num_robots}. Rationale: {spec['rationale']}",
        "meta": {
            "domain": "floortile",
            "enable_numeric": True,
            "enable_conditional": False,
            "enable_durations": False,
            "enable_stochastic": False,
            "max_steps": len(plan) + 5,
            "init_fluents": [["total-cost", [], 0.0]],
        },
        "objects": objects,
        "static_facts": static_facts,
        "init": init,
        "termination": [
            {
                "name": "goal",
                "when": goal_expr,
                "outcome": "success",
                "reward": 1.0,
            }
        ],
        "reference_plan": plan,
    }


def _specs() -> List[dict]:
    return [
        {
            "id": "t01_floortile_easy_a",
            "name": "Floortile Easy A",
            "description": "Paint a 3x3 checkerboard with two robots.",
            "rationale": "Smallest IPC-style grid; two robots split columns without crossing.",
            "rows": 3,
            "cols": 3,
            "robots": 2,
        },
        {
            "id": "t02_floortile_easy_b",
            "name": "Floortile Easy B",
            "description": "Paint a 3x4 checkerboard with two robots.",
            "rationale": "Adds a column while keeping two-robot column segments.",
            "rows": 3,
            "cols": 4,
            "robots": 2,
        },
        {
            "id": "t03_floortile_medium_a",
            "name": "Floortile Medium A",
            "description": "Paint a 4x4 checkerboard with three robots.",
            "rationale": "Moderate grid size with three column segments.",
            "rows": 4,
            "cols": 4,
            "robots": 3,
        },
        {
            "id": "t04_floortile_medium_b",
            "name": "Floortile Medium B",
            "description": "Paint a 4x5 checkerboard with three robots.",
            "rationale": "Wider grid with three segments and longer vertical runs.",
            "rows": 4,
            "cols": 5,
            "robots": 3,
        },
        {
            "id": "t05_floortile_hard_a",
            "name": "Floortile Hard A",
            "description": "Paint a 6x6 checkerboard with four robots.",
            "rationale": "Larger grid with four segments and many paint passes.",
            "rows": 6,
            "cols": 6,
            "robots": 4,
        },
        {
            "id": "t06_floortile_hard_b",
            "name": "Floortile Hard B",
            "description": "Paint a 7x7 checkerboard with four robots.",
            "rationale": "Largest grid in the set with long column segments.",
            "rows": 7,
            "cols": 7,
            "robots": 4,
        },
    ]


def _write_task(out_dir: Path, spec: dict) -> Path:
    task = _build_task(spec)
    path = out_dir / f"{spec['id']}.yaml"
    path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IPC-style Floortile tasks.")
    parser.add_argument(
        "--out-dir",
        default="src/sere/assets/tasks/floortile",
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
