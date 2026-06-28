"""Load the central dependency catalog used by the CLI and templates."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).with_name("dependency_catalog.toml")


@dataclass(frozen=True, slots=True)
class DependencyCatalog:
    default_python: str
    core: tuple[str, ...]
    profiles: dict[str, tuple[str, ...]]
    tracking: dict[str, tuple[str, ...]]
    data_versioning: dict[str, tuple[str, ...]]
    groups: dict[str, tuple[str, ...]]


def _package_sections(raw: dict[str, Any], section: str) -> dict[str, tuple[str, ...]]:
    try:
        return {name: tuple(values["packages"]) for name, values in raw[section].items()}
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid dependency catalog section: {section}") from exc


@lru_cache(maxsize=1)
def load_catalog(path: Path = CATALOG_PATH) -> DependencyCatalog:
    """Load and minimally validate the dependency catalog."""
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    try:
        groups = {name: tuple(packages) for name, packages in raw["groups"].items()}
        catalog = DependencyCatalog(
            default_python=raw["settings"]["default_python"],
            core=tuple(raw["dependencies"]["core"]),
            profiles=_package_sections(raw, "profiles"),
            tracking=_package_sections(raw, "tracking"),
            data_versioning=_package_sections(raw, "data-versioning"),
            groups=groups,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid dependency catalog structure") from exc

    if not catalog.profiles or "dev" not in catalog.groups or "notebooks" not in catalog.groups:
        raise ValueError("Dependency catalog requires profiles, dev, and notebooks sections")
    return catalog


DEPENDENCIES = load_catalog()
