"""Tests for task loading and the per-task run loop."""

import pytest

from harness.runner import load_task, run_task


class _FakeClient:
    """Stands in for DeepSeekClient without making any network call."""

    model = "fake-model"

    def __init__(self, generate_code=None):
        self._generate_code = generate_code

    def generate_code(self, prompt, language, temperature=0.2):
        return self._generate_code(prompt, language)


def test_load_task_requires_all_fields(tmp_path):
    incomplete = tmp_path / "bad.yaml"
    incomplete.write_text("id: bad\nlanguage: python\n")  # missing prompt/test_code
    with pytest.raises(ValueError, match="missing required field"):
        load_task(incomplete)


def test_run_task_survives_an_api_failure_instead_of_raising():
    """Regression test: run_task used to call client.generate_code with no
    try/except, so a single failed API call (rate limit, insufficient
    balance, network blip) would crash run_all and silently drop every
    remaining task in the batch. It must now record the failure and
    return a normal TaskResult."""

    def boom(prompt, language):
        raise RuntimeError("simulated API failure")

    task = {"id": "demo", "language": "python", "prompt": "irrelevant", "test_code": "irrelevant"}
    result = run_task(task, _FakeClient(generate_code=boom))

    assert result.passed is False
    assert "simulated API failure" in result.error
    assert result.task_id == "demo"
