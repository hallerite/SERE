"""Tests for the RML rover navigation environment."""
import math
import pytest

from sere.rml.terrain import TerrainGrid, RoverProblem, CLEAR, ROCK, SAND
from sere.rml.rml_parser import parse_rml_plan
from sere.rml.simulator import validate_plan, format_plan_feedback
from sere.rml.problems import list_problems, load_problem, PROBLEMS
from sere.rml.rml_env import AgenticRMLEnv


# ── Parser ──────────────────────────────────────────────────────────────

class TestParser:
    def test_basic(self):
        xml = '<rover-plan><waypoint x="10" y="20" /><waypoint x="30" y="40" /></rover-plan>'
        wps = parse_rml_plan(xml)
        assert wps == [(10.0, 20.0), (30.0, 40.0)]

    def test_empty_plan(self):
        wps = parse_rml_plan("<rover-plan></rover-plan>")
        assert wps == []

    def test_bad_xml(self):
        with pytest.raises(ValueError, match="Invalid XML"):
            parse_rml_plan("<rover-plan><broken")

    def test_wrong_root(self):
        with pytest.raises(ValueError, match="Expected <rover-plan>"):
            parse_rml_plan("<plan><waypoint x='1' y='2' /></plan>")

    def test_missing_attr(self):
        with pytest.raises(ValueError, match="missing x or y"):
            parse_rml_plan('<rover-plan><waypoint x="1" /></rover-plan>')

    def test_non_numeric(self):
        with pytest.raises(ValueError, match="Non-numeric"):
            parse_rml_plan('<rover-plan><waypoint x="abc" y="2" /></rover-plan>')

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Empty plan"):
            parse_rml_plan("")


# ── Terrain ─────────────────────────────────────────────────────────────

class TestTerrain:
    @pytest.fixture
    def simple_terrain(self):
        return TerrainGrid(
            width=4, height=4, cell_size=10.0,
            terrain=[[CLEAR]*4 for _ in range(4)],
            elevation=[[0.0]*4 for _ in range(4)],
        )

    def test_in_bounds(self, simple_terrain):
        assert simple_terrain.in_bounds(0, 0)
        assert simple_terrain.in_bounds(39, 39)
        assert not simple_terrain.in_bounds(-1, 0)
        assert not simple_terrain.in_bounds(0, 40)

    def test_cell_at(self, simple_terrain):
        assert simple_terrain.cell_at(5.0, 15.0) == (0, 1)
        assert simple_terrain.cell_at(35.0, 35.0) == (3, 3)

    def test_slope_flat(self, simple_terrain):
        assert simple_terrain.slope_between(0, 0, 10, 10) == 0.0

    def test_slope_uphill(self):
        elev = [[0.0]*4 for _ in range(4)]
        elev[0][0] = 0.0
        elev[3][3] = 30.0
        t = TerrainGrid(4, 4, 10.0, [[CLEAR]*4 for _ in range(4)], elev)
        slope = t.slope_between(5.0, 5.0, 35.0, 35.0)
        assert slope > 0

    def test_cells_along_path(self, simple_terrain):
        cells = simple_terrain.cells_along_path(5.0, 5.0, 35.0, 35.0)
        assert (0, 0) in cells
        assert (3, 3) in cells
        assert len(cells) >= 4

    def test_render_ascii(self, simple_terrain):
        text = simple_terrain.render_ascii((5.0, 5.0), (35.0, 35.0))
        assert "S" in text
        assert "G" in text


# ── Simulator ───────────────────────────────────────────────────────────

class TestSimulator:
    @pytest.fixture
    def flat_problem(self):
        return RoverProblem(
            "test", TerrainGrid(
                10, 10, 10.0,
                [[CLEAR]*10 for _ in range(10)],
                [[0.0]*10 for _ in range(10)],
            ),
            start=(5.0, 5.0), goal=(85.0, 85.0),
            goal_tolerance=10.0, max_slope=30.0, max_energy=500.0,
        )

    def test_valid_plan(self, flat_problem):
        wps = [(45.0, 45.0), (85.0, 85.0)]
        r = validate_plan(flat_problem, wps)
        assert r.success
        assert r.goal_distance < flat_problem.goal_tolerance

    def test_empty_plan(self, flat_problem):
        r = validate_plan(flat_problem, [])
        assert not r.success
        assert "no waypoints" in r.error.lower()

    def test_out_of_bounds(self, flat_problem):
        r = validate_plan(flat_problem, [(200.0, 200.0)])
        assert not r.success
        assert "out of bounds" in r.error.lower()

    def test_obstacle_collision(self):
        g = [[CLEAR]*5 for _ in range(5)]
        g[2][2] = ROCK
        p = RoverProblem(
            "blocked", TerrainGrid(5, 5, 10.0, g, [[0.0]*5 for _ in range(5)]),
            start=(5.0, 5.0), goal=(45.0, 45.0),
            goal_tolerance=5.0, max_slope=30.0, max_energy=200.0,
        )
        # Path goes through rock at (2,2)
        r = validate_plan(p, [(25.0, 25.0), (45.0, 45.0)])
        assert not r.success
        assert "impassable" in r.error.lower() or "crosses" in r.error.lower()

    def test_goal_not_reached(self, flat_problem):
        r = validate_plan(flat_problem, [(30.0, 30.0)])
        assert not r.success
        assert r.goal_distance > flat_problem.goal_tolerance

    def test_energy_exceeded(self):
        p = RoverProblem(
            "low_energy", TerrainGrid(
                10, 10, 10.0,
                [[CLEAR]*10 for _ in range(10)],
                [[0.0]*10 for _ in range(10)],
            ),
            start=(5.0, 5.0), goal=(85.0, 85.0),
            goal_tolerance=10.0, max_slope=30.0,
            max_energy=10.0,  # very low
        )
        r = validate_plan(p, [(85.0, 85.0)])
        assert not r.success
        assert "energy" in r.error.lower()

    def test_slope_exceeded(self):
        elev = [[0.0]*5 for _ in range(5)]
        elev[4][4] = 100.0  # steep
        p = RoverProblem(
            "steep", TerrainGrid(5, 5, 10.0, [[CLEAR]*5 for _ in range(5)], elev),
            start=(5.0, 5.0), goal=(45.0, 45.0),
            goal_tolerance=5.0, max_slope=5.0, max_energy=500.0,
        )
        r = validate_plan(p, [(45.0, 45.0)])
        assert not r.success
        assert "slope" in r.error.lower()

    def test_format_success(self, flat_problem):
        r = validate_plan(flat_problem, [(45.0, 45.0), (85.0, 85.0)])
        text = format_plan_feedback(r)
        assert "valid" in text.lower()

    def test_format_failure(self, flat_problem):
        r = validate_plan(flat_problem, [])
        text = format_plan_feedback(r)
        assert "no waypoints" in text.lower()


# ── Problems ────────────────────────────────────────────────────────────

class TestProblems:
    def test_all_problems_load(self):
        for name in list_problems():
            p = load_problem(name)
            assert p.name
            assert p.terrain.width > 0
            assert p.terrain.height > 0

    def test_all_problems_render(self):
        for name in list_problems():
            p = load_problem(name)
            text = p.render_problem()
            assert "TERRAIN MAP" in text
            assert "ELEVATION" in text

    @pytest.mark.parametrize("name", list(PROBLEMS.keys()))
    def test_start_goal_in_bounds(self, name):
        p = load_problem(name)
        assert p.terrain.in_bounds(*p.start)
        assert p.terrain.in_bounds(*p.goal)

    @pytest.mark.parametrize("name", list(PROBLEMS.keys()))
    def test_start_not_on_obstacle(self, name):
        p = load_problem(name)
        assert p.terrain.terrain_at(*p.start) not in ("#", "*")

    def test_unknown_problem(self):
        with pytest.raises(ValueError):
            load_problem("nonexistent")


# ── Agentic Env ─────────────────────────────────────────────────────────

class TestAgenticEnv:
    @pytest.fixture
    def env(self):
        p = load_problem("open_field")
        e = AgenticRMLEnv(problem=p, max_attempts=3)
        yield e
        e.cleanup()

    def test_workspace_created(self, env):
        ws = env.workspace
        assert (ws / "problem.txt").exists()
        assert (ws / "plan.xml").exists()

    def test_read_file(self, env):
        text, done = env.handle_tool_call("read_file", {"path": "problem.txt"})
        assert "ROVER NAVIGATION" in text
        assert not done

    def test_write_and_read(self, env):
        env.handle_tool_call("write_file", {
            "path": "plan.xml",
            "content": '<rover-plan><waypoint x="65" y="65" /></rover-plan>',
        })
        text, _ = env.handle_tool_call("read_file", {"path": "plan.xml"})
        assert "65" in text

    def test_readonly_problem(self, env):
        text, _ = env.handle_tool_call("write_file", {
            "path": "problem.txt", "content": "hacked",
        })
        assert "read-only" in text.lower()

    def test_partial_validate(self, env):
        env.handle_tool_call("write_file", {
            "path": "plan.xml",
            "content": '<rover-plan><waypoint x="30" y="30" /><waypoint x="65" y="65" /></rover-plan>',
        })
        text, done = env.handle_tool_call("validate", {"up_to_waypoint": 1})
        assert not done
        assert env.attempts == 0  # partial doesn't count

    def test_full_validate_success(self, env):
        # Avoid rocks at cells (3,2), (3,3), (5,5) — go via left edge
        env.handle_tool_call("write_file", {
            "path": "plan.xml",
            "content": '<rover-plan><waypoint x="5" y="65" /><waypoint x="65" y="65" /></rover-plan>',
        })
        text, done = env.handle_tool_call("validate", {})
        assert env.attempts == 1
        assert env.solved
        assert done

    def test_max_attempts(self, env):
        # Write a plan that fails (goal too far)
        env.handle_tool_call("write_file", {
            "path": "plan.xml",
            "content": '<rover-plan><waypoint x="10" y="10" /></rover-plan>',
        })
        for i in range(3):
            text, done = env.handle_tool_call("validate", {})
        assert env.attempts == 3
        assert done
        assert not env.solved

    def test_simulate(self, env):
        env.handle_tool_call("write_file", {
            "path": "plan.xml",
            "content": '<rover-plan><waypoint x="35" y="35" /></rover-plan>',
        })
        text, done = env.handle_tool_call("simulate", {})
        assert "TERRAIN WITH ROUTE" in text
        assert not done

    def test_str_replace(self, env):
        env.handle_tool_call("write_file", {
            "path": "plan.xml",
            "content": '<rover-plan><waypoint x="10" y="10" /></rover-plan>',
        })
        env.handle_tool_call("str_replace", {
            "path": "plan.xml",
            "old_str": 'x="10" y="10"',
            "new_str": 'x="65" y="65"',
        })
        text, _ = env.handle_tool_call("read_file", {"path": "plan.xml"})
        assert 'x="65"' in text

    def test_bash(self, env):
        text, done = env.handle_tool_call("bash", {"command": "ls"})
        assert "plan.xml" in text
        assert "problem.txt" in text
        assert not done

    def test_unknown_tool(self, env):
        text, done = env.handle_tool_call("fly_rover", {})
        assert "Unknown tool" in text

    def test_system_prompt(self, env):
        sp = env.system_prompt()
        assert "rover" in sp.lower()
        assert "3" in sp  # max_attempts
