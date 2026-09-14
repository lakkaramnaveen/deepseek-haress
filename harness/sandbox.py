"""Runs model-generated code inside an isolated Docker container.

Isolation: no network access by default, capped CPU/memory, a fresh
throwaway container per run, and a wall-clock timeout enforced from the
host. This is meant to contain buggy/untrusted generated code during
testing, not to withstand a deliberately adversarial attacker.

Two entry points:
  * run_in_sandbox         - single solution + test file (per-task harness)
  * run_project_in_sandbox - a whole directory tree (codebase translation)
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

# Images used to *verify* a translated codebase. A superset of IMAGES
# above (which only covers the two languages the per-task harness scores)
# since a translation target can be any of these.
PROJECT_IMAGES = {
    **IMAGES,
    "typescript": "node:20-slim",
    "go": "golang:1.22-slim",
    "ruby": "ruby:3.3-slim",
    "java": "eclipse-temurin:21-jdk",
}

DEFAULT_TIMEOUT_SECS = 20
DEFAULT_PROJECT_TIMEOUT_SECS = 180


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


def _error_result(message: str) -> SandboxResult:
    return SandboxResult(
        passed=False, stdout="", stderr="", returncode=-1, timed_out=False, error=message
    )


def _check_prerequisites(language: str, images: dict[str, str]) -> SandboxResult | None:
    """Shared preflight for both sandbox entry points: confirm Docker is
    available and the language has an image. Returns an error
    SandboxResult if something's wrong, or None if it's fine to proceed."""
    if not docker_available():
        return _error_result(
            "Docker is not installed or not on PATH. Install Docker "
            "Desktop (https://www.docker.com/products/docker-desktop/) "
            "to run the sandboxed test bed."
        )
    if language not in images:
        return _error_result(
            f"Unsupported language: {language!r}. Supported: {list(images)}"
        )
    return None


def _run_docker(docker_cmd: list[str], timeout: int) -> SandboxResult:
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


def run_in_sandbox(
    language: str,
    solution_code: str,
    test_code: str,
    timeout: int = DEFAULT_TIMEOUT_SECS,
) -> SandboxResult:
    error = _check_prerequisites(language, IMAGES)
    if error:
        return error

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

        return _run_docker(docker_cmd, timeout)


def run_project_in_sandbox(
    language: str,
    project_dir: Path,
    run_cmd: str,
    install_cmd: str | None = None,
    timeout: int = DEFAULT_PROJECT_TIMEOUT_SECS,
    network: bool = False,
    memory: str = "1g",
    cpus: str = "2",
) -> SandboxResult:
    """Run an arbitrary command against a whole project directory.

    Unlike run_in_sandbox (a single solution + test file), this mounts an
    entire translated codebase read-write and runs a shell command inside
    it -- e.g. `pip install -r requirements.txt && pytest` or
    `npm install && npm test`. Network is off by default (matching the
    rest of this harness's isolation policy); pass network=True when the
    project needs to install dependencies.
    """
    error = _check_prerequisites(language, PROJECT_IMAGES)
    if error:
        return error

    project_dir = Path(project_dir).resolve()
    full_cmd = f"{install_cmd} && {run_cmd}" if install_cmd else run_cmd

    docker_cmd = [
        "docker", "run", "--rm",
        *([] if network else ["--network", "none"]),
        "--memory", memory,
        "--cpus", cpus,
        "-v", f"{project_dir}:/workspace",
        "-w", "/workspace",
        PROJECT_IMAGES[language],
        "sh", "-c", full_cmd,
    ]

    return _run_docker(docker_cmd, timeout)
