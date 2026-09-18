"""Tests for the whole-codebase translation agent.

translate_codebase is exercised end-to-end against a fake client (no
network) and real temp-directory I/O (no Docker) -- these are the parts
of the pipeline that don't require live model access, and where the
bugs fixed in this module actually lived: filename collisions, nested
--out directories, and stale-file overwrites.
"""

from pathlib import Path

import pytest

from harness.translate_agent import (
    _build_manifest,
    _map_target_path,
    discover_other_files,
    discover_source_files,
    translate_codebase,
)


class _FakeClient:
    """Translates by uppercasing the source and recording the manifest
    each call was given, so tests can assert on both without any
    network access."""

    def __init__(self):
        self.calls = []

    def translate_file(self, source_code, source_language, target_language, rel_path, manifest="", temperature=0.1):
        self.calls.append({"rel_path": rel_path, "manifest": manifest})
        return {"code": source_code.upper(), "raw_response": source_code.upper(), "usage": None}


# --- _map_target_path -------------------------------------------------


def test_map_target_path_uses_language_output_ext_by_default():
    assert _map_target_path(Path("app.py"), ".js") == Path("app.js")


@pytest.mark.parametrize(
    "source_name,output_ext,expected_name",
    [
        ("Vector.h", ".cpp", "Vector.h"),      # header kept, not collapsed to .cpp
        ("Button.jsx", ".js", "Button.jsx"),   # jsx kept, not collapsed to .js
        ("worker.mjs", ".js", "worker.mjs"),   # ESM semantics kept
    ],
)
def test_map_target_path_preserves_meaningful_extensions(source_name, output_ext, expected_name):
    assert _map_target_path(Path(source_name), output_ext) == Path(expected_name)


# --- _build_manifest / collision detection -----------------------------


def test_build_manifest_lists_every_rename():
    rels = [Path("a.py"), Path("pkg/b.py")]
    targets = [Path("a.js"), Path("pkg/b.js")]
    manifest, collisions = _build_manifest(rels, targets)
    assert "a.py -> a.js" in manifest
    assert "pkg/b.py -> pkg/b.js" in manifest
    assert collisions == {}


def test_build_manifest_flags_a_genuine_collision():
    """Two distinct files mapping to the same target (e.g. Foo.cc and
    Foo.cpp both -> Foo.cpp) must be reported, not silently merged."""
    rels = [Path("Foo.cc"), Path("Foo.cpp")]
    targets = [Path("Foo.cpp"), Path("Foo.cpp")]
    manifest, collisions = _build_manifest(rels, targets)
    assert Path("Foo.cpp") in collisions
    assert set(collisions[Path("Foo.cpp")]) == {Path("Foo.cc"), Path("Foo.cpp")}
    assert "Foo.cc" not in manifest and "Foo.cpp" not in manifest


# --- discover_source_files / discover_other_files ----------------------


def test_discover_source_files_filters_by_extension_and_skips_excluded_dirs(tmp_path):
    (tmp_path / "main.py").write_text("print(1)")
    (tmp_path / "README.md").write_text("docs")
    excluded = tmp_path / "node_modules"
    excluded.mkdir()
    (excluded / "ignored.py").write_text("print(2)")

    found = discover_source_files(tmp_path, "python")
    assert [p.name for p in found] == ["main.py"]


def test_discover_source_files_orders_entry_points_last(tmp_path):
    (tmp_path / "main.py").write_text("")
    (tmp_path / "utils.py").write_text("")

    found = discover_source_files(tmp_path, "python")
    assert [p.name for p in found] == ["utils.py", "main.py"]


def test_discover_source_files_excludes_nested_out_dir(tmp_path):
    (tmp_path / "main.py").write_text("")
    out_dir = tmp_path / "translated"
    out_dir.mkdir()
    (out_dir / "leftover.py").write_text("")  # stale output from a prior run

    found = discover_source_files(tmp_path, "python", exclude_root=out_dir)
    assert [p.name for p in found] == ["main.py"]


def test_discover_other_files_excludes_nested_out_dir(tmp_path):
    (tmp_path / "README.md").write_text("docs")
    out_dir = tmp_path / "translated"
    out_dir.mkdir()
    (out_dir / "leftover.js").write_text("")

    found = discover_other_files(tmp_path, "python", exclude_root=out_dir)
    assert [p.name for p in found] == ["README.md"]


# --- translate_codebase (end-to-end, fake client, real temp dirs) ------


def test_translate_codebase_writes_files_and_uses_full_manifest(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    (src / "shapes").mkdir(parents=True)
    (src / "main.py").write_text("from shapes.rectangle import Rectangle")
    (src / "shapes" / "rectangle.py").write_text("class Rectangle: pass")

    client = _FakeClient()
    report = translate_codebase(client, src, out, "python", "javascript")

    assert report.succeeded == 2
    assert (out / "main.js").read_text() == "FROM SHAPES.RECTANGLE IMPORT RECTANGLE"
    assert (out / "shapes" / "rectangle.js").read_text() == "CLASS RECTANGLE: PASS"

    # Every call -- including the one for the leaf file translated first --
    # must see the FULL project manifest, not just siblings translated so
    # far, so forward references and circular imports resolve correctly.
    for call in client.calls:
        assert "main.py -> main.js" in call["manifest"]
        assert "shapes/rectangle.py -> shapes/rectangle.js" in call["manifest"]


def test_translate_codebase_fails_colliding_files_without_writing_either(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    (src / "Foo.cc").write_text("// impl")
    (src / "Foo.cpp").write_text("// also impl")

    client = _FakeClient()
    report = translate_codebase(client, src, out, "cpp", "cpp")

    assert report.succeeded == 0
    assert report.failed == 2
    assert all("would also be used by" in f.error for f in report.files)
    assert not (out / "Foo.cpp").exists()
    assert client.calls == []  # no API cost spent on files that can't be written anyway


def test_translate_codebase_does_not_let_a_stale_copy_clobber_a_translation(tmp_path):
    """A pre-existing utils.ts sitting next to utils.js (e.g. a partially
    migrated repo) must not overwrite the freshly translated utils.ts."""
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    (src / "utils.js").write_text("module.exports = {}")
    (src / "utils.ts").write_text("// stale hand-written stub, not the translation")

    client = _FakeClient()
    report = translate_codebase(client, src, out, "javascript", "typescript")

    assert report.succeeded == 1
    assert (out / "utils.ts").read_text() == "MODULE.EXPORTS = {}"
    assert report.other_files_copied == 0


def test_translate_codebase_does_not_duplicate_nested_out_dir(tmp_path):
    src = tmp_path
    out = tmp_path / "translated"
    (src / "main.py").write_text("print(1)")

    client = _FakeClient()
    report = translate_codebase(client, src, out, "python", "javascript", copy_other_files=True)

    assert report.succeeded == 1
    assert not (out / "translated").exists()
