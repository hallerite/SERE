#!/usr/bin/env python3
"""
Generate IPC-style Goldminer tasks (2D grid with rocks and guaranteed path).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


Pos = Tuple[int, int]


def _loc(r: int, c: int) -> str:
    return f"f{r}-{c}"


def _neighbors(rows: int, cols: int) -> List[Tuple[Pos, Pos]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
                edges.append(((r, c + 1), (r, c)))
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
                edges.append(((r + 1, c), (r, c)))
    return edges


def _carve_path(rows: int, cols: int, rng: random.Random, gold: Pos) -> List[Pos]:
    r, c = gold
    path = [(r, c)]
    while c > 1:
        if rows > 1 and rng.random() < 0.4:
            if r == 0:
                r = 1
            elif r == rows - 1:
                r = rows - 2
            else:
                r += rng.choice([-1, 1])
            path.append((r, c))
        c -= 1
        path.append((r, c))
    return path


def _grid_positions(rows: int, cols: int) -> List[Pos]:
    return [(r, c) for r in range(rows) for c in range(cols)]


def _build_plan(
    rows: int,
    cols: int,
    *,
    robot: Pos,
    supply: Pos,
    gold: Pos,
    rock_cells: Set[Pos],
    path_from_gold: Sequence[Pos],
) -> List[str]:
    plan: List[str] = []
    robot_loc = robot
    supply_loc = supply

    def move_along_column(start: Pos, end_row: int) -> Pos:
        r, c = start
        if r == end_row:
            return (r, c)
        step = 1 if end_row > r else -1
        while r != end_row:
            nxt = r + step
            plan.append(f"(move {_loc(r, c)} {_loc(nxt, c)})")
            r = nxt
        return (r, c)

    robot_loc = move_along_column(robot_loc, supply_loc[0])
    plan.append(f"(pickup-laser {_loc(*supply_loc)})")

    path_to_gold = list(reversed(path_from_gold))
    path_to_adj = path_to_gold[:-1]  # exclude gold itself

    entry = path_to_adj[0]
    robot_loc = move_along_column(robot_loc, entry[0])

    for cell in path_to_adj:
        if cell in rock_cells:
            plan.append(f"(fire-laser {_loc(*robot_loc)} {_loc(*cell)})")
        plan.append(f"(move {_loc(*robot_loc)} {_loc(*cell)})")
        robot_loc = cell

    for cell in reversed(path_to_adj[:-1]):
        plan.append(f"(move {_loc(*robot_loc)} {_loc(*cell)})")
        robot_loc = cell

    if robot_loc[1] != 0:
        plan.append(f"(move {_loc(*robot_loc)} {_loc(robot_loc[0], 0)})")
        robot_loc = (robot_loc[0], 0)

    robot_loc = move_along_column(robot_loc, supply_loc[0])
    plan.append(f"(putdown-laser {_loc(*supply_loc)})")
    plan.append(f"(pickup-bomb {_loc(*supply_loc)})")

    robot_loc = move_along_column(robot_loc, entry[0])
    if robot_loc[1] != 0:
        raise ValueError("Expected to be in column 0 before re-entering path")

    for cell in path_to_adj:
        plan.append(f"(move {_loc(*robot_loc)} {_loc(*cell)})")
        robot_loc = cell

    plan.append(f"(detonate-bomb {_loc(*robot_loc)} {_loc(*gold)})")
    plan.append(f"(move {_loc(*robot_loc)} {_loc(*gold)})")
    plan.append(f"(pick-gold {_loc(*gold)})")

    return plan


def _build_task(spec: dict) -> dict:
    rows = spec["rows"]
    cols = spec["cols"]
    rng = random.Random(spec["seed"])

    robot_row = rng.randrange(rows)
    supply_row = rng.randrange(rows)
    if rows > 1:
        while supply_row == robot_row:
            supply_row = rng.randrange(rows)
    gold_row = rng.randrange(rows)

    robot = (robot_row, 0)
    supply = (supply_row, 0)
    gold = (gold_row, cols - 1)

    soft: Set[Pos] = set()
    hard: Set[Pos] = set()

    for r, c in _grid_positions(rows, cols):
        if c == 0:
            continue
        if (r, c) == gold:
            continue
        if rng.random() < 0.5:
            soft.add((r, c))
        else:
            hard.add((r, c))

    path_from_gold = _carve_path(rows, cols, rng, gold)
    for cell in path_from_gold:
        if cell == gold:
            continue
        soft.add(cell)
        hard.discard(cell)

    rock_cells = set(soft) | set(hard)

    plan = _build_plan(
        rows,
        cols,
        robot=robot,
        supply=supply,
        gold=gold,
        rock_cells=rock_cells,
        path_from_gold=path_from_gold,
    )

    static_facts = []
    for a, b in _neighbors(rows, cols):
        static_facts.append(f"(connected {_loc(*a)} {_loc(*b)})")

    init = ["(arm-empty)"]
    init.append(f"(robot-at {_loc(*robot)})")
    init.append(f"(bomb-at {_loc(*supply)})")
    init.append(f"(laser-at {_loc(*supply)})")

    for r, c in _grid_positions(rows, cols):
        loc = _loc(r, c)
        if (r, c) == robot or (r, c) == supply:
            init.append(f"(clear {loc})")
            continue
        if (r, c) == gold:
            init.append(f"(gold-at {loc})")
            init.append(f"(soft-rock-at {loc})")
            continue
        if (r, c) in soft:
            init.append(f"(soft-rock-at {loc})")
        elif (r, c) in hard:
            init.append(f"(hard-rock-at {loc})")
        else:
            init.append(f"(clear {loc})")

    objects = { _loc(r, c): "loc" for r, c in _grid_positions(rows, cols) }

    max_steps = max(10, len(plan) + 5)

    return {
        "id": spec["id"],
        "name": spec["name"],
        "description": (
            f"{spec['description']} Settings: rows={rows}, cols={cols}, seed={spec['seed']}."
            f" Rationale: {spec['rationale']}"
        ),
        "meta": {
            "domain": "goldminer",
            "enable_numeric": False,
            "enable_conditional": False,
            "enable_durations": False,
            "enable_stochastic": False,
            "max_steps": max_steps,
        },
        "objects": objects,
        "static_facts": static_facts,
        "init": init,
        "termination": [
            {"name": "goal", "when": "(holds-gold)", "outcome": "success", "reward": 1.0}
        ],
        "reference_plan": plan,
    }


def _specs() -> List[dict]:
    return [
        {
            "id": "t01_goldminer_easy_a",
            "name": "Goldminer Easy A",
            "description": "Reach the gold in a small 2D mine.",
            "rationale": "Small grid with a short carved path.",
            "rows": 2,
            "cols": 3,
            "seed": 1,
        },
        {
            "id": "t02_goldminer_easy_b",
            "name": "Goldminer Easy B",
            "description": "Clear a slightly wider grid to retrieve the gold.",
            "rationale": "Adds one column for longer navigation.",
            "rows": 2,
            "cols": 4,
            "seed": 2,
        },
        {
            "id": "t03_goldminer_medium_a",
            "name": "Goldminer Medium A",
            "description": "Navigate a medium grid with more rock variation.",
            "rationale": "More rows increase possible rock placements.",
            "rows": 3,
            "cols": 4,
            "seed": 3,
        },
        {
            "id": "t04_goldminer_medium_b",
            "name": "Goldminer Medium B",
            "description": "Traverse a larger mine with longer horizontal distance.",
            "rationale": "Longer columns require more clearing before the return trip.",
            "rows": 3,
            "cols": 5,
            "seed": 4,
        },
        {
            "id": "t05_goldminer_hard_a",
            "name": "Goldminer Hard A",
            "description": "Solve a large grid with a longer carved path.",
            "rationale": "More cells increase the clearing workload.",
            "rows": 4,
            "cols": 6,
            "seed": 5,
        },
        {
            "id": "t06_goldminer_hard_b",
            "name": "Goldminer Hard B",
            "description": "Navigate the largest grid in the set.",
            "rationale": "Max size and rock variety push the strategy length.",
            "rows": 5,
            "cols": 7,
            "seed": 6,
        },
    ]


def _write_task(out_dir: Path, spec: dict) -> Path:
    task = _build_task(spec)
    path = out_dir / f"{spec['id']}.yaml"
    path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IPC-style Goldminer tasks.")
    parser.add_argument(
        "--out-dir",
        default="src/sere/assets/tasks/goldminer",
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
