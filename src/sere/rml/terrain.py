"""Terrain grid for rover navigation problems."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Terrain types
CLEAR = "."
ROCK = "#"
SAND = "~"
CRATER = "*"

IMPASSABLE = {ROCK, CRATER}
ENERGY_MULTIPLIER = {
    CLEAR: 1.0,
    SAND: 2.0,
    ROCK: float("inf"),
    CRATER: float("inf"),
}


@dataclass
class TerrainGrid:
    """2D terrain grid with elevation."""
    width: int
    height: int
    cell_size: float  # meters per cell
    terrain: List[List[str]]  # terrain[row][col]
    elevation: List[List[float]]  # elevation[row][col] in meters

    def in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x < self.width * self.cell_size and 0 <= y < self.height * self.cell_size

    def cell_at(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coords to grid cell (col, row)."""
        col = min(int(x / self.cell_size), self.width - 1)
        row = min(int(y / self.cell_size), self.height - 1)
        return max(0, col), max(0, row)

    def terrain_at(self, x: float, y: float) -> str:
        col, row = self.cell_at(x, y)
        return self.terrain[row][col]

    def elevation_at(self, x: float, y: float) -> float:
        col, row = self.cell_at(x, y)
        return self.elevation[row][col]

    def slope_between(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Slope angle in degrees between two points."""
        dx, dy = x2 - x1, y2 - y1
        horiz = math.hypot(dx, dy)
        if horiz < 1e-6:
            return 0.0
        dz = self.elevation_at(x2, y2) - self.elevation_at(x1, y1)
        return math.degrees(math.atan2(abs(dz), horiz))

    def cells_along_path(
        self, x1: float, y1: float, x2: float, y2: float,
    ) -> List[Tuple[int, int]]:
        """Grid cells the straight-line path passes through (DDA)."""
        c1, r1 = self.cell_at(x1, y1)
        c2, r2 = self.cell_at(x2, y2)
        dc, dr = c2 - c1, r2 - r1
        steps = max(abs(dc), abs(dr), 1)
        cells: List[Tuple[int, int]] = []
        for i in range(steps + 1):
            t = i / steps
            c = round(c1 + dc * t)
            r = round(r1 + dr * t)
            if not cells or cells[-1] != (c, r):
                cells.append((c, r))
        return cells

    def render_ascii(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]] = None,
    ) -> str:
        """Render terrain as ASCII with optional waypoint overlay."""
        grid = [row[:] for row in self.terrain]

        sc, sr = self.cell_at(*start)
        gc, gr = self.cell_at(*goal)
        grid[sr][sc] = "S"
        grid[gr][gc] = "G"

        if waypoints:
            for i, (wx, wy) in enumerate(waypoints):
                wc, wr = self.cell_at(wx, wy)
                if grid[wr][wc] not in ("S", "G"):
                    grid[wr][wc] = str(i % 10)

        lines = ["    " + " ".join(f"{c:>2}" for c in range(self.width))]
        for r in range(self.height):
            lines.append(f"{r:>2}  " + "  ".join(grid[r]))
        return "\n".join(lines)

    def render_elevation(self) -> str:
        lines = ["    " + " ".join(f"{c:>4}" for c in range(self.width))]
        for r in range(self.height):
            vals = " ".join(f"{self.elevation[r][c]:4.0f}" for c in range(self.width))
            lines.append(f"{r:>2}  {vals}")
        return "\n".join(lines)


@dataclass
class RoverProblem:
    """A rover navigation problem."""
    name: str
    terrain: TerrainGrid
    start: Tuple[float, float]
    goal: Tuple[float, float]
    goal_tolerance: float
    max_slope: float  # degrees
    max_energy: float  # watt-hours
    energy_rate: float = 1.0  # Wh/m on clear terrain

    def render_problem(self) -> str:
        t = self.terrain
        lines = [
            f"=== ROVER NAVIGATION: {self.name} ===",
            "",
            f"Grid: {t.width}x{t.height} cells, {t.cell_size:.0f}m per cell "
            f"({t.width * t.cell_size:.0f}m x {t.height * t.cell_size:.0f}m)",
            f"Start: ({self.start[0]:.1f}, {self.start[1]:.1f})",
            f"Goal:  ({self.goal[0]:.1f}, {self.goal[1]:.1f})",
            f"Goal tolerance: {self.goal_tolerance:.1f}m",
            "",
            "=== CONSTRAINTS ===",
            f"Max slope: {self.max_slope:.0f} degrees",
            f"Max energy: {self.max_energy:.0f} Wh",
            f"Energy cost: {self.energy_rate:.1f} Wh/m (clear), "
            f"{self.energy_rate * ENERGY_MULTIPLIER[SAND]:.1f} Wh/m (sand)",
            "Rock (#) and crater (*) are impassable.",
            "",
            "=== TERRAIN MAP ===",
            "Legend: . = clear, # = rock, ~ = sand, * = crater, S = start, G = goal",
            f"Coords: x right (cols), y down (rows). World = cell * {t.cell_size:.0f}m",
            "",
            t.render_ascii(self.start, self.goal),
            "",
            "=== ELEVATION (m) ===",
            t.render_elevation(),
            "",
            "=== SOLUTION FORMAT ===",
            "Write waypoints in plan.xml:",
            "<rover-plan>",
            '  <waypoint x="10.0" y="5.0" />',
            '  <waypoint x="20.0" y="15.0" />',
            "</rover-plan>",
            "",
            "Rover drives from start to each waypoint in sequence.",
            "Final waypoint must be within goal tolerance.",
        ]
        return "\n".join(lines)
