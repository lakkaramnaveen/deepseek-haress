"""Runs model-generated code inside an isolated Docker container.

Isolation: no network access, capped CPU/memory, a fresh throwaway
container per run, and a wall-clock timeout enforced from the host.
This is meant to contain buggy/untrusted generated code during testing,
not to withstand a deliberately adversarial attacker.
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

IMAGES = {
    "python": "python:3.11-slim",
    "javascript": "node:20-slim",
}

RUN_CMD = {
    "python": ["python3", "test.py"],
    "javascript": ["node", "test.js"],
}

DEFAULT_TIMEOUT_SECS = 20


@dataclass
class SandboxResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    error: str | None = None


def docker_available() -> bool:
    return shutil.which("docker") is not None


def run_in_sandbox(
    language: str,
    solution_code: str,
    test_code: str,
    timeout: int = DEFAULT_TIMEOUT_SECS,
) -> SandboxResult:
    if not docker_available():
        return SandboxResult(
            passed=False,
            stdout="",
            stderr="",
            returncode=-1,
            timed_out=False,
            error=(
                "Docker is not installed or not on PATH. Install Docker "
                "Desktop (https://www.docker.com/products/docker-desktop/) "
                "to run the sandboxed test bed."
            ),
        )

    if language not in IMAGES:
        return SandboxResult(
            passed=False,
            stdout="",
            stderr="",
            returncode=-1,
            timed_out=False,
            error=f"Unsupported language: {language!r}. Supported: {list(IMAGES)}",
        )

    with tempfile.TemporaryDirectory(prefix="dsh-sandbox-") as tmpdir:
        workdir = Path(tmpdir)
        if language == "python":
            (workdir / "solution.py").write_text(solution_code)
            (workdir / "test.py").write_text(test_code)
        elif language == "javascript":
            (workdir / "solution.js").write_text(solution_code)
            (workdir / "test.js").write_text(test_code)

        docker_cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "256m",
            "--cpus", "1",
            "-v", f"{workdir}:/workspace",
            "-w", "/workspace",
            IMAGES[language],
            *RUN_CMD[language],
        ]

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                passed=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                returncode=-1,
                timed_out=True,
                error=f"Execution exceeded {timeout}s timeout",
            )

        return SandboxResult(
            passed=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            timed_out=False,
        )
