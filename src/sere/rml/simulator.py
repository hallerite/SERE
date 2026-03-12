"""Rover path simulator — validates waypoint plans against terrain constraints."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sere.rml.terrain import ENERGY_MULTIPLIER, IMPASSABLE, RoverProblem


@dataclass
class WaypointResult:
    success: bool
    index: int
    from_pos: Tuple[float, float]
    to_pos: Tuple[float, float]
    distance: float = 0.0
    energy: float = 0.0
    slope: float = 0.0
    error: Optional[str] = None


@dataclass
class PlanResult:
    success: bool
    waypoints_ok: int
    total_waypoints: int
    total_distance: float = 0.0
    total_energy: float = 0.0
    final_pos: Tuple[float, float] = (0.0, 0.0)
    goal_distance: float = float("inf")
    error: Optional[str] = None
    failed: Optional[WaypointResult] = None


def _validate_transition(
    problem: RoverProblem,
    from_pos: Tuple[float, float],
    to_pos: Tuple[float, float],
    idx: int,
    cumulative_energy: float,
) -> WaypointResult:
    """Validate driving from one position to the next."""
    t = problem.terrain
    x2, y2 = to_pos

    # Bounds
    if not t.in_bounds(x2, y2):
        return WaypointResult(
            success=False, index=idx, from_pos=from_pos, to_pos=to_pos,
            error=f"Waypoint {idx} ({x2:.1f}, {y2:.1f}) out of bounds "
                  f"(area: {t.width * t.cell_size:.0f}x{t.height * t.cell_size:.0f}m)",
        )

    # Destination terrain
    if t.terrain_at(x2, y2) in IMPASSABLE:
        return WaypointResult(
            success=False, index=idx, from_pos=from_pos, to_pos=to_pos,
            error=f"Waypoint {idx} ({x2:.1f}, {y2:.1f}) on impassable terrain "
                  f"({t.terrain_at(x2, y2)})",
        )

    # Path crosses impassable cells
    cells = t.cells_along_path(*from_pos, *to_pos)
    for col, row in cells:
        ct = t.terrain[row][col]
        if ct in IMPASSABLE:
            cx = (col + 0.5) * t.cell_size
            cy = (row + 0.5) * t.cell_size
            return WaypointResult(
                success=False, index=idx, from_pos=from_pos, to_pos=to_pos,
                error=f"Path to waypoint {idx} crosses {ct} at cell ({col},{row}) "
                      f"≈ ({cx:.0f}m, {cy:.0f}m)",
            )

    # Slope
    slope = t.slope_between(*from_pos, *to_pos)
    if slope > problem.max_slope:
        return WaypointResult(
            success=False, index=idx, from_pos=from_pos, to_pos=to_pos,
            slope=slope,
            error=f"Slope to waypoint {idx} is {slope:.1f}° (max {problem.max_slope:.0f}°)",
        )

    # Distance and energy
    dist = math.hypot(to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
    total_mult = sum(
        ENERGY_MULTIPLIER.get(t.terrain[r][c], 1.0) for c, r in cells
    )
    avg_mult = total_mult / max(len(cells), 1)
    energy = dist * problem.energy_rate * avg_mult

    if cumulative_energy + energy > problem.max_energy:
        return WaypointResult(
            success=False, index=idx, from_pos=from_pos, to_pos=to_pos,
            distance=dist, energy=energy, slope=slope,
            error=f"Energy exceeded at waypoint {idx}: "
                  f"{cumulative_energy + energy:.0f} Wh (max {problem.max_energy:.0f})",
        )

    return WaypointResult(
        success=True, index=idx, from_pos=from_pos, to_pos=to_pos,
        distance=dist, energy=energy, slope=slope,
    )


def validate_plan(
    problem: RoverProblem,
    waypoints: List[Tuple[float, float]],
) -> PlanResult:
    """Validate a complete waypoint plan."""
    if not waypoints:
        return PlanResult(
            success=False, waypoints_ok=0, total_waypoints=0,
            final_pos=problem.start,
            goal_distance=math.dist(problem.start, problem.goal),
            error="Plan has no waypoints",
        )

    pos = problem.start
    total_dist = 0.0
    total_energy = 0.0

    for i, wp in enumerate(waypoints):
        wr = _validate_transition(problem, pos, wp, i, total_energy)
        if not wr.success:
            return PlanResult(
                success=False, waypoints_ok=i, total_waypoints=len(waypoints),
                total_distance=total_dist, total_energy=total_energy,
                final_pos=pos,
                goal_distance=math.dist(pos, problem.goal),
                error=wr.error, failed=wr,
            )
        total_dist += wr.distance
        total_energy += wr.energy
        pos = wp

    goal_dist = math.dist(pos, problem.goal)
    ok = goal_dist <= problem.goal_tolerance

    return PlanResult(
        success=ok,
        waypoints_ok=len(waypoints),
        total_waypoints=len(waypoints),
        total_distance=total_dist,
        total_energy=total_energy,
        final_pos=pos,
        goal_distance=goal_dist,
        error=None if ok else
            f"Final pos ({pos[0]:.1f}, {pos[1]:.1f}) is {goal_dist:.1f}m from goal "
            f"({problem.goal[0]:.1f}, {problem.goal[1]:.1f}), "
            f"need within {problem.goal_tolerance:.1f}m",
    )


def format_plan_feedback(result: PlanResult) -> str:
    """Human-readable validation feedback."""
    if result.success:
        return (
            f"Plan valid! Rover reached the goal.\n"
            f"  Waypoints: {result.waypoints_ok}\n"
            f"  Distance:  {result.total_distance:.1f}m\n"
            f"  Energy:    {result.total_energy:.0f} Wh\n"
            f"  Goal dist: {result.goal_distance:.1f}m"
        )

    lines = []
    if result.failed:
        lines.append(
            f"Failed at waypoint {result.failed.index} of {result.total_waypoints}."
        )
    else:
        lines.append(f"All {result.waypoints_ok} waypoints OK, but goal not reached.")
    lines.append(f"Error: {result.error}")
    lines.append(f"\nRover state:")
    lines.append(f"  Position: ({result.final_pos[0]:.1f}, {result.final_pos[1]:.1f})")
    lines.append(f"  Distance: {result.total_distance:.1f}m")
    lines.append(f"  Energy:   {result.total_energy:.0f} Wh")
    lines.append(f"  Goal dist: {result.goal_distance:.1f}m")
    return "\n".join(lines)
