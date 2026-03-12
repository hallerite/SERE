"""Hand-crafted rover navigation problems of varying difficulty."""
from __future__ import annotations

import math
from typing import Dict, List, Callable

from sere.rml.terrain import (
    CLEAR, CRATER, ROCK, SAND, RoverProblem, TerrainGrid,
)


def _grid(w: int, h: int, fill: str = CLEAR) -> List[List[str]]:
    return [[fill] * w for _ in range(h)]


def _elev(w: int, h: int, base: float = 0.0) -> List[List[float]]:
    return [[base] * w for _ in range(h)]


def _hill(elev: List[List[float]], cx: int, cy: int, r: float, h: float):
    for row in range(len(elev)):
        for col in range(len(elev[0])):
            d2 = (col - cx) ** 2 + (row - cy) ** 2
            elev[row][col] += h * math.exp(-d2 / (2 * r * r))


# ── p01: Open Field (trivial) ──────────────────────────────────────────
def make_open_field() -> RoverProblem:
    w, h, cs = 8, 8, 10.0
    g = _grid(w, h)
    e = _elev(w, h)
    g[2][3] = ROCK
    g[3][3] = ROCK
    g[5][5] = ROCK
    return RoverProblem(
        "Open Field",
        TerrainGrid(w, h, cs, g, e),
        start=(5.0, 5.0), goal=(65.0, 65.0),
        goal_tolerance=10.0, max_slope=30.0, max_energy=200.0,
    )


# ── p02: Wall Gap (easy) ───────────────────────────────────────────────
def make_wall_gap() -> RoverProblem:
    w, h, cs = 12, 10, 10.0
    g = _grid(w, h)
    e = _elev(w, h)
    for c in range(w):
        if c != 8:
            g[4][c] = ROCK
            g[5][c] = ROCK
    return RoverProblem(
        "Wall Gap",
        TerrainGrid(w, h, cs, g, e),
        start=(15.0, 15.0), goal=(15.0, 85.0),
        goal_tolerance=10.0, max_slope=30.0, max_energy=300.0,
    )


# ── p03: Sandy Detour (medium) ─────────────────────────────────────────
def make_sandy_detour() -> RoverProblem:
    w, h, cs = 12, 12, 10.0
    g = _grid(w, h)
    e = _elev(w, h)
    # Sand block in center
    for r in range(3, 9):
        for c in range(3, 9):
            g[r][c] = SAND
    # Clear corridor: top rows and right edge
    for c in range(w):
        g[0][c] = CLEAR
        g[1][c] = CLEAR
    for r in range(h):
        g[r][10] = CLEAR
        g[r][11] = CLEAR
    return RoverProblem(
        "Sandy Detour",
        TerrainGrid(w, h, cs, g, e),
        start=(5.0, 5.0), goal=(55.0, 105.0),
        goal_tolerance=10.0, max_slope=30.0,
        max_energy=300.0,  # tight — sand path may bust budget
    )


# ── p04: Ridge Crossing (medium-hard) ──────────────────────────────────
def make_ridge_crossing() -> RoverProblem:
    w, h, cs = 15, 12, 10.0
    g = _grid(w, h)
    e = _elev(w, h)
    # East-west ridge at rows 5-6, low pass near col 11
    for r in range(h):
        for c in range(w):
            ridge_d = min(abs(r - 5), abs(r - 6))
            ridge_h = max(0.0, 40.0 - ridge_d * 15.0)
            # Pass at col 11 cuts ridge height by ~80%
            pass_f = 1.0 - 0.8 * math.exp(-((c - 11) ** 2) / 4.0)
            e[r][c] = ridge_h * pass_f
    return RoverProblem(
        "Ridge Crossing",
        TerrainGrid(w, h, cs, g, e),
        start=(15.0, 5.0), goal=(15.0, 105.0),
        goal_tolerance=10.0, max_slope=15.0,  # tight
        max_energy=400.0,
    )


# ── p05: Crater Field (hard) ───────────────────────────────────────────
def make_crater_field() -> RoverProblem:
    w, h, cs = 15, 15, 10.0
    g = _grid(w, h)
    e = _elev(w, h)
    obs = [
        (2, 1), (3, 1), (5, 2), (6, 2), (7, 2),
        (1, 3), (2, 3), (9, 3), (10, 3),
        (4, 5), (5, 5), (6, 5), (8, 5),
        (2, 7), (3, 7), (7, 7), (8, 7), (11, 7),
        (5, 9), (6, 9), (9, 9), (10, 9),
        (3, 11), (7, 11), (8, 11), (11, 11), (12, 11),
        (5, 13), (9, 13), (10, 13),
    ]
    for c, r in obs:
        if 0 <= r < h and 0 <= c < w:
            g[r][c] = CRATER if (c + r) % 3 == 0 else ROCK
    _hill(e, 7, 7, 5.0, 15.0)
    return RoverProblem(
        "Crater Field",
        TerrainGrid(w, h, cs, g, e),
        start=(0.0, 0.0), goal=(140.0, 140.0),
        goal_tolerance=10.0, max_slope=20.0, max_energy=500.0,
    )


# ── p06: Canyon Run (hard) ─────────────────────────────────────────────
def make_canyon_run() -> RoverProblem:
    w, h, cs = 20, 10, 10.0
    g = _grid(w, h)
    e = _elev(w, h)
    # Canyon walls with gaps
    for c in range(w):
        if c not in (5, 12, 18):
            g[1][c] = ROCK
            g[2][c] = ROCK
        if c not in (3, 10, 16):
            g[7][c] = ROCK
            g[8][c] = ROCK
    # Internal obstacles
    g[4][6] = ROCK; g[4][7] = ROCK
    g[5][13] = ROCK; g[5][14] = ROCK
    g[3][9] = CRATER; g[6][4] = CRATER
    # Sand zone
    for c in range(8, 12):
        g[4][c] = SAND
        g[5][c] = SAND
    # Eastward elevation rise + central hill
    for r in range(h):
        for c in range(w):
            e[r][c] = c * 1.5
    _hill(e, 10, 5, 4.0, 20.0)
    return RoverProblem(
        "Canyon Run",
        TerrainGrid(w, h, cs, g, e),
        start=(5.0, 45.0), goal=(190.0, 45.0),
        goal_tolerance=10.0, max_slope=20.0, max_energy=600.0,
    )


# ── Registry ────────────────────────────────────────────────────────────

PROBLEMS: Dict[str, Callable[[], RoverProblem]] = {
    "open_field": make_open_field,
    "wall_gap": make_wall_gap,
    "sandy_detour": make_sandy_detour,
    "ridge_crossing": make_ridge_crossing,
    "crater_field": make_crater_field,
    "canyon_run": make_canyon_run,
}


def list_problems() -> List[str]:
    return list(PROBLEMS.keys())


def load_problem(name: str) -> RoverProblem:
    if name not in PROBLEMS:
        raise ValueError(f"Unknown: {name}. Available: {list(PROBLEMS.keys())}")
    return PROBLEMS[name]()
