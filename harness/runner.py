"""Loads tasks, asks DeepSeek to solve them, and scores results in the sandbox."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .client import DeepSeekClient
from .sandbox import run_in_sandbox

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class TaskResult:
    task_id: str
    language: str
    passed: bool
    error: str | None
    model: str
    duration_secs: float
    generated_code: str
    stdout: str
    stderr: str


def load_task(path: Path) -> dict:
    with open(path) as f:
        task = yaml.safe_load(f)
    for field in ("id", "language", "prompt", "test_code"):
        if field not in task:
            raise ValueError(f"{path}: task is missing required field '{field}'")
    return task


def load_all_tasks() -> list[dict]:
    return [load_task(p) for p in sorted(TASKS_DIR.glob("*.yaml"))]


def run_task(task: dict, client: DeepSeekClient) -> TaskResult:
    start = time.monotonic()
    generation = client.generate_code(task["prompt"], task["language"])
    code = generation["code"]

    sandbox_result = run_in_sandbox(
        language=task["language"],
        solution_code=code,
        test_code=task["test_code"],
        timeout=task.get("timeout", 20),
    )
    duration = time.monotonic() - start

    return TaskResult(
        task_id=task["id"],
        language=task["language"],
        passed=sandbox_result.passed,
        error=sandbox_result.error,
        model=client.model,
        duration_secs=round(duration, 2),
        generated_code=code,
        stdout=sandbox_result.stdout,
        stderr=sandbox_result.stderr,
    )


def save_result(result: TaskResult) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{ts}-{result.task_id}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))
    return out_path


def run_all(client: DeepSeekClient, task_filter: str | None = None) -> list[TaskResult]:
    tasks = load_all_tasks()
    if task_filter:
        tasks = [t for t in tasks if t["id"] == task_filter]
        if not tasks:
            raise ValueError(f"No task found with id '{task_filter}'")

    results = []
    for task in tasks:
        print(f"Running task '{task['id']}' ({task['language']})...")
        result = run_task(task, client)
        save_result(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  -> {status} in {result.duration_secs}s")
        if not result.passed and result.error:
            print(f"     {result.error}")
        results.append(result)
    return results
