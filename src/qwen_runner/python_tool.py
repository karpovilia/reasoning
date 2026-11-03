from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


class PythonToolExecutor:
    """Sandboxed python code-interpreter tool, plug-compatible with
    WikipediaToolExecutor (openai_tools property + execute_tool_call).

    Sandbox: subprocess `python -I -c <code>` in a fresh tmpdir with resource
    limits (CPU/AS/FSIZE/NOFILE), env cleared to PATH only. Never raises: every
    outcome is a dict with stdout/stderr/exit_code/wall_s/timeout.

    Config via env:
      TOOL_PYTHON     — interpreter to run tool code with (default: sys.executable);
                        point it at a thin venv with sympy/numpy installed.
      TOOL_TIMEOUT_S  — wall-clock timeout per call, seconds (default 10).
    """

    OUTPUT_CHAR_LIMIT = 2000
    # Pro-tool nudge (PLAN.md L1: prompt tuning is the sanctioned remedy when
    # call share is low). An anti-tool nudge ("think first") is forbidden (R-1).
    system_hint = (
        "Use the python tool whenever a calculation, simplification, or "
        "enumeration is needed — it is more reliable than computing by hand."
    )

    def __init__(self) -> None:
        self.python_bin = os.getenv("TOOL_PYTHON") or sys.executable
        self.timeout_s = float(os.getenv("TOOL_TIMEOUT_S", "10"))

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": (
                        "Execute python code and return its stdout/stderr "
                        f"(truncated to {self.OUTPUT_CHAR_LIMIT} characters). "
                        "Use print() to output results. Available libraries: "
                        "math, fractions, sympy, numpy. No network access."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. Print anything you need to see.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def execute_tool_call(self, tool_name: str, arguments_json: str) -> dict[str, Any]:
        if tool_name != "python":
            raise ValueError(f"Unsupported tool: {tool_name}")
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            return self._error_result(f"invalid tool arguments JSON: {exc}")
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return self._error_result("missing code argument")
        return self.run_code(code)

    def run_code(self, code: str) -> dict[str, Any]:
        tmpdir = tempfile.mkdtemp(prefix="pytool_")
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [self.python_bin, "-I", "-c", code],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_s,
                cwd=tmpdir,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                preexec_fn=self._apply_rlimits,
            )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = self._decode(exc.stdout)
            stderr = self._decode(exc.stderr) + f"\n[timeout after {self.timeout_s:.0f}s]"
            exit_code = -1
        except OSError as exc:
            stdout, stderr, exit_code = "", f"sandbox launch failed: {exc}", -1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        wall_s = time.monotonic() - started
        return {
            "stdout": stdout[: self.OUTPUT_CHAR_LIMIT],
            "stderr": stderr[: self.OUTPUT_CHAR_LIMIT],
            "exit_code": exit_code,
            "wall_s": round(wall_s, 3),
            "timeout": timed_out,
        }

    @staticmethod
    def _decode(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    @staticmethod
    def _apply_rlimits() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    def _error_result(self, message: str) -> dict[str, Any]:
        return {
            "stdout": "",
            "stderr": message,
            "exit_code": -1,
            "wall_s": 0.0,
            "timeout": False,
            "error": message,
        }
