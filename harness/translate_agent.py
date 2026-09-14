"""Whole-codebase source-to-source translation agent.

Walks a source directory, translates every source file to a target
language one at a time via DeepSeekClient.translate_file, and writes the
result to an output directory with the same relative layout. Files are
translated in a stable order (leaf-ish files before "main"/"index"-style
entry points), and every prompt is given the *complete* project manifest
(every file's original path -> new path) up front. That mapping only
depends on file discovery, never on translated content, so precomputing
it means even a file translated early can correctly reference a sibling
that imports it later, or half of a circular import pair.

This is a best-effort translation, not a compiler: always review the
output, and use `verify_codebase` (backed by harness/sandbox.py) to
actually execute the translated project before trusting it.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from .client import DeepSeekClient
from .sandbox import SandboxResult, run_project_in_sandbox

# Extensions searched for when discovering source files of a given
# language, and the extension normally used for translated output files.
LANGUAGE_EXTENSIONS: dict[str, dict] = {
    "python": {"source_exts": [".py"], "output_ext": ".py"},
    "javascript": {"source_exts": [".js", ".jsx", ".mjs", ".cjs"], "output_ext": ".js"},
    "typescript": {"source_exts": [".ts", ".tsx"], "output_ext": ".ts"},
    "go": {"source_exts": [".go"], "output_ext": ".go"},
    "ruby": {"source_exts": [".rb"], "output_ext": ".rb"},
    "java": {"source_exts": [".java"], "output_ext": ".java"},
    "rust": {"source_exts": [".rs"], "output_ext": ".rs"},
    "php": {"source_exts": [".php"], "output_ext": ".php"},
    "c": {"source_exts": [".c", ".h"], "output_ext": ".c"},
    "cpp": {"source_exts": [".cpp", ".cc", ".hpp", ".h"], "output_ext": ".cpp"},
    "csharp": {"source_exts": [".cs"], "output_ext": ".cs"},
}

# Source extensions that must keep their own extension in the output
# rather than collapsing onto their language's default output_ext above.
# Without this, e.g. Vector.h and Vector.cpp would both map to
# Vector.cpp (one silently overwriting the other on disk), and
# translating a .mjs file to "javascript" would lose the ESM-vs-CJS
# module semantics that .mjs/.cjs specifically signal to Node.
PRESERVED_EXTENSIONS = {
    ".h": ".h",
    ".hpp": ".hpp",
    ".jsx": ".jsx",
    ".tsx": ".tsx",
    ".mjs": ".mjs",
    ".cjs": ".cjs",
}

# Directories never walked into when discovering files. Deliberately
# narrow: e.g. "bin" is excluded from some other tools' defaults, but
# it's real hand-written source in some ecosystems (Rails' bin/rails),
# so we don't blanket-exclude it here -- extension filtering in
# discover_source_files already skips compiled binaries that live there.
EXCLUDE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "target", ".mypy_cache", ".pytest_cache", ".idea",
    ".vscode", "vendor",
}

ENTRY_POINT_STEMS = {"main", "index", "app", "cli", "server", "__main__"}


@dataclass
class FileTranslation:
    source_path: str
    target_path: str | None
    ok: bool
    error: str | None = None
    chars_in: int = 0
    chars_out: int = 0
    duration_secs: float = 0.0


@dataclass
class TranslationReport:
    source_language: str
    target_language: str
    src_dir: str
    out_dir: str
    files: list[FileTranslation] = field(default_factory=list)
    other_files_copied: int = 0

    @property
    def succeeded(self) -> int:
        return sum(1 for f in self.files if f.ok)

    @property
    def failed(self) -> int:
        return sum(1 for f in self.files if not f.ok)


def _language_config(language: str) -> dict:
    if language not in LANGUAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported language: {language!r}. Supported: "
            f"{list(LANGUAGE_EXTENSIONS)}"
        )
    return LANGUAGE_EXTENSIONS[language]


def _iter_project_files(src_dir: Path, exclude_root: Path | None = None):
    """Yield every plain file under src_dir, skipping EXCLUDE_DIRS and
    (if given) anything under exclude_root.

    exclude_root matters when --out is nested inside --src (e.g. `--src .
    --out ./translated`): without it, files this same run just wrote to
    out_dir would be walked again as if they were part of the source,
    both re-translating them and duplicating them into a nested copy.
    """
    exclude_root = exclude_root.resolve() if exclude_root else None
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if exclude_root is not None and path.resolve().is_relative_to(exclude_root):
            continue
        yield path


def discover_source_files(
    src_dir: Path, language: str, exclude_root: Path | None = None
) -> list[Path]:
    """Find every source file of `language` under src_dir, in a stable,
    dependency-friendly order (files most likely to be *imported by*
    others come before files most likely to *import* others -- purely
    for readability of translation order; correctness of cross-file
    references comes from the full manifest, not from this ordering)."""
    exts = set(_language_config(language)["source_exts"])
    src_dir = Path(src_dir)
    matches = [p for p in _iter_project_files(src_dir, exclude_root) if p.suffix in exts]

    def sort_key(p: Path):
        rel = p.relative_to(src_dir)
        is_entry = p.stem.lower() in ENTRY_POINT_STEMS
        return (is_entry, len(rel.parts), str(rel))

    return sorted(matches, key=sort_key)


def discover_other_files(
    src_dir: Path, language: str, exclude_root: Path | None = None
) -> list[Path]:
    """Non-source files (assets, docs, configs) worth carrying over
    verbatim into the translated project."""
    exts = set(_language_config(language)["source_exts"])
    src_dir = Path(src_dir)
    return [p for p in _iter_project_files(src_dir, exclude_root) if p.suffix not in exts]


def _map_target_path(rel_path: Path, output_ext: str) -> Path:
    preserved = PRESERVED_EXTENSIONS.get(rel_path.suffix)
    return rel_path.with_suffix(preserved if preserved else output_ext)


def _build_manifest(rels: list[Path], target_rels: list[Path]) -> tuple[str, dict[Path, list[Path]]]:
    """Build the full "original -> new path" manifest text, and report any
    collisions where two different source files would map to the same
    target path (see PRESERVED_EXTENSIONS' docstring for the common
    causes). Colliding entries are left out of the manifest text itself,
    since none of them can safely be written."""
    by_target: dict[Path, list[Path]] = {}
    for rel, target_rel in zip(rels, target_rels):
        by_target.setdefault(target_rel, []).append(rel)
    collisions = {target: srcs for target, srcs in by_target.items() if len(srcs) > 1}

    manifest = "\n".join(
        f"{rel} -> {target_rel}"
        for rel, target_rel in zip(rels, target_rels)
        if target_rel not in collisions
    )
    return manifest, collisions


def translate_codebase(
    client: DeepSeekClient,
    src_dir: str | Path,
    out_dir: str | Path,
    source_language: str,
    target_language: str,
    copy_other_files: bool = True,
    on_progress=None,
) -> TranslationReport:
    """Translate every source_language file under src_dir into
    target_language, writing the result under out_dir with the same
    relative directory structure.

    on_progress, if given, is called as on_progress(index, total, rel_path)
    before each file is translated.
    """
    src_dir = Path(src_dir).resolve()
    out_dir = Path(out_dir).resolve()
    _language_config(source_language)
    output_ext = _language_config(target_language)["output_ext"]

    if not src_dir.is_dir():
        raise ValueError(f"src_dir does not exist or is not a directory: {src_dir}")

    files = discover_source_files(src_dir, source_language, exclude_root=out_dir)
    rels = [p.relative_to(src_dir) for p in files]
    target_rels = [_map_target_path(rel, output_ext) for rel in rels]
    manifest, collisions = _build_manifest(rels, target_rels)

    report = TranslationReport(
        source_language=source_language,
        target_language=target_language,
        src_dir=str(src_dir),
        out_dir=str(out_dir),
    )

    for i, (path, rel, target_rel) in enumerate(zip(files, rels, target_rels)):
        if on_progress:
            on_progress(i + 1, len(files), str(rel))

        if target_rel in collisions:
            others = ", ".join(str(s) for s in collisions[target_rel] if s != rel)
            report.files.append(
                FileTranslation(
                    source_path=str(rel),
                    target_path=None,
                    ok=False,
                    error=(
                        f"target path {target_rel} would also be used by "
                        f"{others}; rename one of the source files, or "
                        "translate them in separate runs"
                    ),
                )
            )
            continue

        start = time.monotonic()
        try:
            source_code = path.read_text()
            result = client.translate_file(
                source_code=source_code,
                source_language=source_language,
                target_language=target_language,
                rel_path=str(rel),
                manifest=manifest,
            )
            code = result["code"]
            target_path = out_dir / target_rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(code)

            report.files.append(
                FileTranslation(
                    source_path=str(rel),
                    target_path=str(target_rel),
                    ok=True,
                    chars_in=len(source_code),
                    chars_out=len(code),
                    duration_secs=round(time.monotonic() - start, 2),
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't abort the run
            report.files.append(
                FileTranslation(
                    source_path=str(rel),
                    target_path=None,
                    ok=False,
                    error=str(exc),
                    duration_secs=round(time.monotonic() - start, 2),
                )
            )

    if copy_other_files:
        # Never let a verbatim copy of an original file clobber a path a
        # successful translation already wrote -- e.g. a project that
        # (unusually) already has a same-named file in the target
        # language sitting next to the one being translated.
        translated_targets = {Path(f.target_path) for f in report.files if f.ok and f.target_path}
        for path in discover_other_files(src_dir, source_language, exclude_root=out_dir):
            rel = path.relative_to(src_dir)
            if rel in translated_targets:
                continue
            target_path = out_dir / rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(path.read_bytes())
            report.other_files_copied += 1

    return report


def verify_codebase(
    language: str,
    project_dir: str | Path,
    run_cmd: str,
    install_cmd: str | None = None,
    timeout: int = 180,
    network: bool = False,
) -> SandboxResult:
    """Execute a translated project inside the Docker sandbox to confirm
    it actually runs, e.g.:

        verify_codebase("javascript", out_dir, run_cmd="npm test",
                         install_cmd="npm install", network=True)
    """
    return run_project_in_sandbox(
        language=language,
        project_dir=Path(project_dir),
        run_cmd=run_cmd,
        install_cmd=install_cmd,
        timeout=timeout,
        network=network,
    )
