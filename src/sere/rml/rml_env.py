"""Agentic sandbox for rover navigation planning (RML)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sere.rml.rml_parser import parse_rml_plan
from sere.rml.simulator import format_plan_feedback, validate_plan
from sere.rml.terrain import RoverProblem

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace a string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_str": {"type": "string", "description": "Text to find"},
                    "new_str": {"type": "string", "description": "Replacement"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate",
            "description": (
                "Validate plan.xml. Without args → full submission (counts as attempt). "
                "With up_to_waypoint=N → partial check (free)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "up_to_waypoint": {
                        "type": "integer",
                        "description": "Check only the first N waypoints (free)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate",
            "description": "Simulate plan.xml and show rover state + route on map.",
            "parameters": {
                "type": "object",
                "properties": {
                    "up_to_waypoint": {
                        "type": "integer",
                        "description": "Simulate only the first N waypoints",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command"},
                },
                "required": ["command"],
            },
        },
    },
]


@dataclass
class AgenticRMLEnv:
    """Agentic sandbox for rover waypoint planning."""

    problem: RoverProblem
    max_attempts: int = 8
    bash_timeout: int = 10

    # Mutable state
    attempts: int = 0
    solved: bool = False
    _workspace: Optional[Path] = field(default=None, repr=False)

    @property
    def workspace(self) -> Path:
        if self._workspace is None:
            self._workspace = Path(tempfile.mkdtemp(prefix="rml_"))
            self._setup_workspace()
        return self._workspace

    def _setup_workspace(self):
        ws = self._workspace
        assert ws is not None
        (ws / "problem.txt").write_text(self.problem.render_problem())
        (ws / "plan.xml").write_text(
            "<rover-plan>\n  <!-- Add waypoints here -->\n</rover-plan>\n"
        )

    def _enforce_readonly(self):
        ws = self.workspace
        expected = self.problem.render_problem()
        p = ws / "problem.txt"
        if not p.exists() or p.read_text() != expected:
            p.write_text(expected)

    @property
    def tool_schemas(self) -> List[Dict]:
        return TOOL_SCHEMAS

    def system_prompt(self) -> str:
        return (
            "You are a rover navigation planner.\n\n"
            "Workspace:\n"
            "  problem.txt — terrain map, constraints, goal (read-only)\n"
            "  plan.xml    — write your waypoint plan here\n\n"
            "Tools: read_file, write_file, str_replace, validate, simulate, bash\n\n"
            "Solution format (plan.xml):\n"
            "<rover-plan>\n"
            '  <waypoint x=\"10.0\" y=\"5.0\" />\n'
            '  <waypoint x=\"20.0\" y=\"15.0\" />\n'
            "</rover-plan>\n\n"
            "Rover drives from start to each waypoint in order.\n"
            "Final waypoint must be within goal tolerance.\n\n"
            "Workflow:\n"
            "1. Read problem.txt — understand terrain, obstacles, elevation, constraints.\n"
            "2. Plan a route avoiding obstacles, within slope and energy limits.\n"
            "3. Write waypoints to plan.xml.\n"
            "4. validate(up_to_waypoint=N) → partial check (free).\n"
            "5. validate() → full submission (counts as attempt).\n\n"
            f"You have {self.max_attempts} submission attempts."
        )

    def handle_tool_call(
        self, name: str, arguments: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """Process a tool call → (feedback, done)."""
        self._enforce_readonly()
        dispatch = {
            "bash": lambda: self._bash(arguments.get("command", "")),
            "read_file": lambda: self._read_file(arguments.get("path", "")),
            "write_file": lambda: self._write_file(
                arguments.get("path", ""), arguments.get("content", ""),
            ),
            "str_replace": lambda: self._str_replace(
                arguments.get("path", ""),
                arguments.get("old_str", ""),
                arguments.get("new_str", ""),
            ),
            "validate": lambda: self._validate(arguments.get("up_to_waypoint")),
            "simulate": lambda: self._simulate(arguments.get("up_to_waypoint")),
        }
        handler = dispatch.get(name)
        if handler is None:
            return f"Unknown tool: {name}", False
        return handler()

    # ── Tool implementations ────────────────────────────────────────────

    def _bash(self, command: str) -> Tuple[str, bool]:
        if not command:
            return "Error: empty command", False
        try:
            r = subprocess.run(
                ["bash", "-c", command],
                capture_output=True, text=True,
                timeout=self.bash_timeout,
                cwd=str(self.workspace),
            )
            out = (r.stdout + r.stderr).strip()
            return (out[:4000] if out else "(no output)"), False
        except subprocess.TimeoutExpired:
            return f"Timed out after {self.bash_timeout}s", False

    def _read_file(self, path: str) -> Tuple[str, bool]:
        fp = self.workspace / path
        if not fp.exists():
            return f"Not found: {path}", False
        try:
            return fp.read_text()[:8000], False
        except (OSError, UnicodeDecodeError) as e:
            return f"Error: {e}", False

    def _write_file(self, path: str, content: str) -> Tuple[str, bool]:
        if path == "problem.txt":
            return "Error: problem.txt is read-only", False
        fp = self.workspace / path
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content)
            return f"Wrote {len(content)} bytes to {path}", False
        except OSError as e:
            return f"Error: {e}", False

    def _str_replace(
        self, path: str, old_str: str, new_str: str,
    ) -> Tuple[str, bool]:
        if path == "problem.txt":
            return "Error: problem.txt is read-only", False
        fp = self.workspace / path
        if not fp.exists():
            return f"Not found: {path}", False
        try:
            text = fp.read_text()
            if old_str not in text:
                return f"String not found in {path}", False
            n = text.count(old_str)
            fp.write_text(text.replace(old_str, new_str))
            return f"Replaced {n} occurrence(s) in {path}", False
        except (OSError, UnicodeDecodeError) as e:
            return f"Error: {e}", False

    def _validate(self, up_to: Optional[int] = None) -> Tuple[str, bool]:
        try:
            xml = (self.workspace / "plan.xml").read_text()
            wps = parse_rml_plan(xml)
        except (OSError, ValueError) as e:
            return f"Parse error: {e}", False

        if not wps:
            return "No waypoints. Add <waypoint x=\"...\" y=\"...\" />.", False

        is_full = up_to is None or up_to >= len(wps)
        plan = wps if is_full else wps[:up_to]
        result = validate_plan(self.problem, plan)

        if not is_full:
            prefix = f"[Partial check: waypoints 0-{up_to - 1} of {len(wps)}]\n"
            return prefix + format_plan_feedback(result), False

        # Full submission
        self.attempts += 1
        self.solved = result.success
        feedback = (
            f"[SUBMISSION {self.attempts}/{self.max_attempts}]\n"
            + format_plan_feedback(result)
        )
        done = result.success or self.attempts >= self.max_attempts
        if not result.success and self.attempts >= self.max_attempts:
            feedback += f"\n\nMax attempts ({self.max_attempts}) reached."
        return feedback, done

    def _simulate(self, up_to: Optional[int] = None) -> Tuple[str, bool]:
        try:
            xml = (self.workspace / "plan.xml").read_text()
            wps = parse_rml_plan(xml)
        except (OSError, ValueError) as e:
            return f"Parse error: {e}", False

        if not wps:
            return "No waypoints.", False

        if up_to is not None:
            wps = wps[:up_to]

        result = validate_plan(self.problem, wps)
        ok_wps = wps[: result.waypoints_ok] if not result.success else wps

        lines = [
            format_plan_feedback(result),
            "",
            "=== TERRAIN WITH ROUTE ===",
            self.problem.terrain.render_ascii(
                self.problem.start, self.problem.goal, ok_wps,
            ),
        ]
        return "\n".join(lines), False

    def cleanup(self):
        if self._workspace and self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)
