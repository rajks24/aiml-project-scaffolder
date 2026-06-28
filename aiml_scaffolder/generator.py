"""Filesystem generation with safe, observable overwrite behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import ProjectOptions
from .templates import project_files


@dataclass(frozen=True, slots=True)
class ScaffoldResult:
    root: Path
    created: tuple[Path, ...]
    skipped: tuple[Path, ...]


def scaffold(
    options: ProjectOptions, *, force: bool = False, dry_run: bool = False
) -> ScaffoldResult:
    """Create a project, preserving existing files unless ``force`` is set."""
    created: list[Path] = []
    skipped: list[Path] = []
    root = options.root

    for relative_name, content in project_files(options).items():
        relative_path = Path(relative_name)
        destination = root / relative_path
        if destination.exists() and not force:
            skipped.append(relative_path)
            continue
        created.append(relative_path)
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")

    return ScaffoldResult(root=root, created=tuple(created), skipped=tuple(skipped))
