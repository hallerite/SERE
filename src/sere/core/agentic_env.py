"""
miniSWE-style PDDL planning sandbox.

The agent works in a real temp directory with files:
  - domain.pddl   (read-only, restored if modified)
  - problem.pddl   (read-only, restored if modified)
  - plan.pddl      (the agent's solution)

Tools (same as a coding agent):
  - bash(command)                  — run shell commands in the workspace
  - read_file(path)                — read a file
  - write_file(path, content)      — create/overwrite a file
  - str_replace(path, old, new)    — targeted string replacement in a file
  - validate(up_to_step?)          — validate plan.pddl against domain+problem
  - simulate(up_to_step?)          — run plan.pddl up to step N, show world state
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sere.pddl.domain_spec import DomainSpec, Predicate
from sere.core.world_state import WorldState
from sere.core.validator import (
    check_goal,
    format_plan_feedback,
    format_step_error,
    validate_plan,
    validate_step,
)
from sere.core.pddl_env.planning import parse_actions


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the workspace directory. Use for inspecting files, text manipulation, or any shell operation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace. domain.pddl and problem.pddl are read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace a string in a file. The old_str must appear exactly once in the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The exact string to find (must be unique in the file)",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The replacement string",
                    },
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
                "Validate plan.pddl against the domain and problem. "
                "Returns pass/fail with diagnostics. "
                "Optionally validate only the first N steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "up_to_step": {
                        "type": "integer",
                        "description": "Only validate the first N steps. If omitted, validates the full plan (counts as a submission attempt).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate",
            "description": "Run plan.pddl up to a given step and show the resulting world state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "up_to_step": {
                        "type": "integer",
                        "description": "Run the first N steps. If omitted, runs the entire plan.",
                    },
                },
            },
        },
    },
]

READONLY_FILES = {"domain.pddl", "problem.pddl"}


@dataclass
class AgenticPDDLEnv:
    """
    miniSWE-style PDDL sandbox with real filesystem workspace.
    """

    domain: DomainSpec
    init_world: WorldState
    static_facts: Set[Predicate]
    goal_expr: str
    domain_pddl: str
    problem_pddl: str
    problem_name: str = ""

    enable_numeric: bool = False
    enable_conditional: bool = False
    max_attempts: int = 8
    bash_timeout: int = 10

    # Mutable state
    attempts: int = 0
    solved: bool = False
    _workspace: Optional[Path] = field(default=None, repr=False)

    @property
    def workspace(self) -> Path:
        """Lazily create the workspace directory with domain + problem files."""
        if self._workspace is None:
            self._workspace = Path(tempfile.mkdtemp(prefix="sere_sandbox_"))
            (self._workspace / "domain.pddl").write_text(self.domain_pddl)
            (self._workspace / "problem.pddl").write_text(self.problem_pddl)
        return self._workspace

    def cleanup(self):
        """Remove the workspace directory."""
        if self._workspace and self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)
            self._workspace = None

    def system_prompt(self) -> str:
        return (
            "You are a PDDL planning agent working in a sandbox environment.\n\n"
            "Your workspace contains:\n"
            "  domain.pddl   — the planning domain (read-only)\n"
            "  problem.pddl  — the planning problem (read-only)\n"
            "  plan.pddl     — write your solution here\n\n"
            "Tools:\n"
            "  bash(command)                 — run shell commands\n"
            "  read_file(path)               — read a file\n"
            "  write_file(path, content)     — create/overwrite a file\n"
            "  str_replace(path, old, new)   — edit a file\n"
            "  validate(up_to_step?)         — validate plan.pddl against the domain and problem\n"
            "  simulate(up_to_step?)         — run plan.pddl, show world state\n\n"
            "Write your plan as grounded PDDL actions in plan.pddl, one per line:\n"
            "  (action-name arg1 arg2)\n"
            "  (action-name arg1 arg2)\n"
            "  ...\n\n"
            "Workflow:\n"
            "1. Read domain.pddl and problem.pddl to understand the task.\n"
            "2. Write your plan to plan.pddl.\n"
            "3. Use validate(up_to_step=N) to check partial plans (free, no attempt used).\n"
            "4. Call validate() to submit your final plan (counts as an attempt).\n\n"
            f"You have {self.max_attempts} submission attempts."
        )

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return list(TOOL_SCHEMAS)

    def tool_schema(self) -> Dict[str, Any]:
        return TOOL_SCHEMAS[4]  # validate

    def _fresh_world(self) -> WorldState:
        return WorldState(
            domain=self.init_world.domain,
            objects={k: set(v) for k, v in self.init_world.objects.items()},
            facts=set(self.init_world.facts),
            fluents=dict(self.init_world.fluents),
        )

    def _read_plan(self) -> str:
        plan_path = self.workspace / "plan.pddl"
        if not plan_path.exists():
            return ""
        return plan_path.read_text()

    def _parse_plan(self) -> List[Tuple[str, Tuple[str, ...]]]:
        text = self._read_plan()
        if not text.strip():
            raise ValueError("plan.pddl is empty. Write a plan first with write_file.")
        return parse_actions(text)

    def _enforce_readonly(self):
        """Restore read-only files if they were modified."""
        ws = self.workspace
        domain_path = ws / "domain.pddl"
        problem_path = ws / "problem.pddl"
        if not domain_path.exists() or domain_path.read_text() != self.domain_pddl:
            domain_path.write_text(self.domain_pddl)
        if not problem_path.exists() or problem_path.read_text() != self.problem_pddl:
            problem_path.write_text(self.problem_pddl)

    # -----------------------------------------------------------------
    #  Tool dispatch
    # -----------------------------------------------------------------

    def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Tuple[str, bool]:
        if name == "bash":
            result = self._bash(arguments.get("command", ""))
            self._enforce_readonly()
            return result, False
        elif name == "read_file":
            return self._read_file(arguments.get("path", "")), False
        elif name == "write_file":
            return self._write_file(
                arguments.get("path", ""),
                arguments.get("content", ""),
            ), False
        elif name == "str_replace":
            return self._str_replace(
                arguments.get("path", ""),
                arguments.get("old_str", ""),
                arguments.get("new_str", ""),
            ), False
        elif name == "validate":
            return self._validate(arguments.get("up_to_step"))
        elif name == "simulate":
            return self._simulate(arguments.get("up_to_step")), False
        else:
            return f"Unknown tool: {name}", False

    # -----------------------------------------------------------------
    #  Tool implementations
    # -----------------------------------------------------------------

    def _bash(self, command: str) -> str:
        if not command.strip():
            return "Error: empty command"
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.bash_timeout,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr
            if result.returncode != 0 and not output:
                output = f"(exit code {result.returncode})"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.bash_timeout}s"
        except Exception as e:
            return f"Error: {e}"

    def _read_file(self, path: str) -> str:
        path = path.strip().lstrip("/")
        filepath = self.workspace / path
        if not filepath.exists():
            return f"File not found: {path}"
        try:
            # Security: don't read outside workspace
            filepath.resolve().relative_to(self.workspace.resolve())
        except ValueError:
            return f"Access denied: {path}"
        return filepath.read_text()

    def _write_file(self, path: str, content: str) -> str:
        path = path.strip().lstrip("/")
        if path in READONLY_FILES:
            return f"Error: {path} is read-only"
        if not content and not content == "":
            return "Error: no content provided"
        filepath = self.workspace / path
        try:
            filepath.resolve().relative_to(self.workspace.resolve())
        except ValueError:
            return f"Access denied: {path}"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        if path == "plan.pddl":
            try:
                actions = parse_actions(content)
                return f"Wrote {path} ({len(actions)} actions)"
            except ValueError:
                return f"Wrote {path} (warning: could not parse as PDDL actions)"
        return f"Wrote {path}"

    def _str_replace(self, path: str, old_str: str, new_str: str) -> str:
        path = path.strip().lstrip("/")
        if path in READONLY_FILES:
            return f"Error: {path} is read-only"
        filepath = self.workspace / path
        if not filepath.exists():
            return f"File not found: {path}"
        try:
            filepath.resolve().relative_to(self.workspace.resolve())
        except ValueError:
            return f"Access denied: {path}"
        content = filepath.read_text()
        count = content.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {path}"
        if count > 1:
            return f"Error: old_str appears {count} times in {path} (must be unique)"
        content = content.replace(old_str, new_str, 1)
        filepath.write_text(content)
        return f"Replaced in {path}"

    def _validate(self, up_to_step: int | None = None) -> Tuple[str, bool]:
        try:
            full_plan = self._parse_plan()
        except ValueError as e:
            return str(e), False

        # Treat up_to_step >= total steps as a full submission
        is_full = up_to_step is None or up_to_step >= len(full_plan)
        plan = full_plan if is_full else full_plan[:up_to_step]

        result = validate_plan(
            self.domain, self.init_world, self.static_facts,
            self.goal_expr, plan,
            enable_numeric=self.enable_numeric,
            enable_conditional=self.enable_conditional,
        )

        if not is_full:
            return format_plan_feedback(result), False

        # Full submission
        self.attempts += 1
        self.solved = result.success
        feedback = format_plan_feedback(result)

        done = result.success or self.attempts >= self.max_attempts
        if not result.success and self.attempts >= self.max_attempts:
            feedback += f"\n\nMax attempts ({self.max_attempts}) reached."

        return feedback, done

    def _simulate(self, up_to_step: int | None = None) -> str:
        try:
            plan = self._parse_plan()
        except ValueError as e:
            return str(e)

        if up_to_step is not None:
            plan = plan[:up_to_step]

        world = self._fresh_world()

        for i, (name, args) in enumerate(plan):
            result = validate_step(
                self.domain, world, self.static_facts, name, args,
                enable_numeric=self.enable_numeric,
                enable_conditional=self.enable_conditional,
            )
            if not result.success:
                return (
                    format_step_error(result, i) + "\n\n"
                    + _format_state(world, self.static_facts, self.goal_expr, self.enable_numeric)
                )

        return _format_state(
            world, self.static_facts, self.goal_expr, self.enable_numeric,
            prefix=f"State after {len(plan)} steps:",
        )


def _format_state(
    world: WorldState,
    static_facts: Set[Predicate] | None = None,
    goal_expr: str | None = None,
    enable_numeric: bool = False,
    prefix: str = "State:",
) -> str:
    lines = [prefix]
    for pred, args in sorted(world.facts):
        lines.append(f"  ({pred} {' '.join(args)})")
    for (fname, fargs), fval in sorted(world.fluents.items()):
        lines.append(f"  (= ({fname} {' '.join(fargs)}) {fval})")
    if goal_expr and static_facts is not None:
        goal_met = check_goal(world, static_facts, goal_expr, enable_numeric)
        lines.append(f"Goal satisfied: {goal_met}")
    return "\n".join(lines)
