"""Input model and validation for generated projects."""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from pathlib import Path

from .catalog import DEPENDENCIES

PROFILES = tuple(DEPENDENCIES.profiles)
TRACKERS = tuple(DEPENDENCIES.tracking)
DATA_VERSIONING = tuple(DEPENDENCIES.data_versioning)
ENVIRONMENT_MANAGERS = ("uv", "conda", "both")


def package_name(project_name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", project_name).strip("_").lower()
    if value and (value[0].isdigit() or keyword.iskeyword(value)):
        value = f"project_{value}"
    return value


@dataclass(frozen=True, slots=True)
class ProjectOptions:
    project_name: str
    title: str
    author: str
    base_dir: Path
    profiles: tuple[str, ...] = ("general",)
    tracking: str = "local"
    data_versioning: str = "none"
    environment_manager: str = "uv"
    python_version: str = DEPENDENCIES.default_python
    include_ci: bool = True

    def __post_init__(self) -> None:
        name = self.project_name.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError(
                "Project name must start with a letter or number and contain only letters, "
                "numbers, dots, hyphens, or underscores."
            )
        if not package_name(name):
            raise ValueError("Project name must contain at least one letter or number.")
        object.__setattr__(self, "project_name", name)
        normalized_profiles = tuple(dict.fromkeys(self.profiles))
        if not normalized_profiles:
            raise ValueError("Select at least one profile.")
        unknown_profiles = [profile for profile in normalized_profiles if profile not in PROFILES]
        if unknown_profiles:
            raise ValueError(f"Unknown profiles: {', '.join(unknown_profiles)}")
        object.__setattr__(self, "profiles", normalized_profiles)
        if self.tracking not in TRACKERS:
            raise ValueError(f"Unknown tracking backend: {self.tracking}")
        if self.data_versioning not in DATA_VERSIONING:
            raise ValueError(f"Unknown data versioning option: {self.data_versioning}")
        if self.environment_manager not in ENVIRONMENT_MANAGERS:
            raise ValueError(f"Unknown environment manager: {self.environment_manager}")
        if not re.fullmatch(r"3\.(1[1-9]|[2-9][0-9])", self.python_version):
            raise ValueError("Python version must look like 3.12 (minimum: 3.11).")

    @property
    def package_name(self) -> str:
        return package_name(self.project_name)

    @property
    def root(self) -> Path:
        return self.base_dir.expanduser().resolve() / self.project_name
