"""Tests for the sandbox preflight guards.

These test the pure decision logic (docker present? language supported?)
without ever invoking Docker, so they run in any environment.
"""

from harness import sandbox


def test_check_prerequisites_reports_missing_docker(monkeypatch):
    monkeypatch.setattr(sandbox, "docker_available", lambda: False)
    result = sandbox._check_prerequisites("python", sandbox.IMAGES)
    assert result is not None
    assert not result.passed
    assert "Docker is not installed" in result.error


def test_check_prerequisites_reports_unsupported_language(monkeypatch):
    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    result = sandbox._check_prerequisites("cobol", sandbox.IMAGES)
    assert result is not None
    assert not result.passed
    assert "Unsupported language" in result.error


def test_check_prerequisites_passes_for_a_supported_language(monkeypatch):
    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    assert sandbox._check_prerequisites("python", sandbox.IMAGES) is None


def test_project_images_is_a_superset_of_images():
    """PROJECT_IMAGES is derived from IMAGES so the two pinned tags can
    never silently drift apart."""
    for language, image in sandbox.IMAGES.items():
        assert sandbox.PROJECT_IMAGES[language] == image
