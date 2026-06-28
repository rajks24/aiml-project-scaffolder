"""Command-line interface and interactive project wizard."""

from __future__ import annotations

import argparse
import getpass
import sys
from datetime import date
from pathlib import Path

from .catalog import DEPENDENCIES
from .generator import scaffold
from .models import (
    DATA_VERSIONING,
    ENVIRONMENT_MANAGERS,
    PROFILES,
    TRACKERS,
    ProjectOptions,
)


def default_base_dir(source_file: Path | None = None, current_dir: Path | None = None) -> Path:
    """Return the source repository's parent, falling back to the working directory."""
    source_root = (source_file or Path(__file__)).resolve().parent.parent
    if (source_root / "pyproject.toml").is_file() and (source_root / "aiml_scaffolder").is_dir():
        return source_root.parent

    working_dir = (current_dir or Path.cwd()).resolve()
    if (working_dir / "pyproject.toml").is_file() and (working_dir / "aiml_scaffolder").is_dir():
        return working_dir.parent
    return working_dir


def _ask(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def _choose(prompt: str, choices: tuple[str, ...], default: str) -> str:
    print(f"\n{prompt}")
    for index, choice in enumerate(choices, 1):
        marker = " (recommended)" if choice == default else ""
        print(f"  {index}. {choice}{marker}")
    while True:
        answer = input(f"Choose 1-{len(choices)} [{choices.index(default) + 1}]: ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        print("Enter one of the displayed numbers.")


def _choose_many(
    prompt: str, choices: tuple[str, ...], defaults: tuple[str, ...]
) -> tuple[str, ...]:
    print(f"\n{prompt}")
    for index, choice in enumerate(choices, 1):
        marker = " (recommended default)" if choice in defaults else ""
        print(f"  {index}. {choice}{marker}")
    default_numbers = ",".join(str(choices.index(choice) + 1) for choice in defaults)
    while True:
        answer = input(f"Choose one or more, comma-separated [{default_numbers}]: ").strip()
        if not answer:
            return defaults
        parts = [part.strip() for part in answer.split(",") if part.strip()]
        if parts and all(part.isdigit() and 1 <= int(part) <= len(choices) for part in parts):
            return tuple(dict.fromkeys(choices[int(part) - 1] for part in parts))
        print("Enter one or more displayed numbers separated by commas, for example: 1,4,5.")


def _confirm(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{hint}]: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reproducible, production-shaped AI/ML experiment project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("project_name", nargs="?", help="new project folder name")
    parser.add_argument("--title", help="human-readable project title")
    parser.add_argument("--author", help="author or team name")
    parser.add_argument(
        "--path",
        "--base-dir",
        dest="base_dir",
        default=str(default_base_dir()),
        help="directory that will contain the new project",
    )
    parser.add_argument(
        "--profile",
        dest="profiles",
        choices=PROFILES,
        action="append",
        help="composable dependency profile; repeat for multiple profiles",
    )
    parser.add_argument("--tracking", choices=TRACKERS, default="local")
    parser.add_argument("--data-versioning", choices=DATA_VERSIONING, default="none")
    parser.add_argument(
        "--environment",
        "--environment-manager",
        dest="environment_manager",
        choices=ENVIRONMENT_MANAGERS,
        default="uv",
    )
    parser.add_argument("--python", dest="python_version", default=DEPENDENCIES.default_python)
    parser.add_argument("--ci", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interactive", action="store_true", help="run the guided wizard")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="show files without writing")
    return parser


def _options_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ProjectOptions:
    interactive = args.interactive or args.project_name is None
    if interactive and not sys.stdin.isatty():
        parser.error("project_name is required when input is not interactive")

    name = args.project_name
    if interactive:
        print("AIML Project Scaffolder\nAnswer a few questions; Enter accepts the default.\n")
        name = _ask("Project folder name", name or "my-aiml-experiment")
        default_title = name.replace("-", " ").replace("_", " ").title()
        title = _ask("Project title", args.title or default_title)
        author = _ask("Author/team", args.author or getpass.getuser())
        profiles = _choose_many(
            "Experiment profiles", PROFILES, tuple(args.profiles or ("general",))
        )
        tracking = _choose("Experiment tracking", TRACKERS, args.tracking)
        data_versioning = _choose("Data versioning", DATA_VERSIONING, args.data_versioning)
        environment_manager = _choose(
            "Environment manager", ENVIRONMENT_MANAGERS, args.environment_manager
        )
        include_ci = _confirm("Include GitHub Actions CI?", args.ci)
    else:
        assert name is not None
        title = args.title or name.replace("-", " ").replace("_", " ").title()
        author = args.author or getpass.getuser()
        profiles = tuple(args.profiles or ("general",))
        tracking = args.tracking
        data_versioning = args.data_versioning
        environment_manager = args.environment_manager
        include_ci = args.ci

    return ProjectOptions(
        project_name=name,
        title=title,
        author=author,
        base_dir=Path(args.base_dir),
        profiles=profiles,
        tracking=tracking,
        data_versioning=data_versioning,
        environment_manager=environment_manager,
        python_version=args.python_version,
        include_ci=include_ci,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = _options_from_args(args, parser)
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        return 130
    except ValueError as exc:
        parser.error(str(exc))

    result = scaffold(options, force=args.force, dry_run=args.dry_run)
    action = "Would create" if args.dry_run else "Created"
    print(f"\n{action} {len(result.created)} files in {result.root}")
    if result.skipped:
        print(f"Preserved {len(result.skipped)} existing files (use --force to replace them).")
    if args.dry_run:
        for path in result.created:
            print(f"  {path}")
        return 0

    print("\nNext steps:")
    print(f"  cd {result.root}")
    extra_flags = " ".join(f"--extra {profile}" for profile in options.profiles)
    if options.environment_manager == "uv":
        print(f"  uv sync {extra_flags} --group notebooks")
        print("  uv run experiment --config configs/experiment.toml")
        print("  uv run pytest")
    elif options.environment_manager == "conda":
        print("  conda env create --file environment.yml")
        print(f"  conda activate {options.project_name}")
        print("  experiment --config configs/experiment.toml")
        print("  pytest")
    else:
        print(f"  # Choose one: uv sync {extra_flags} --group notebooks")
        print("  #          or conda env create --file environment.yml")
        print("  # See README.md for activation, testing, and Jupyter kernel setup.")
    if options.data_versioning == "dvc":
        dvc = "uv run dvc" if options.environment_manager == "uv" else "dvc"
        print(f"  git init && {dvc} init")
        print(f"  # Add data, remove its .gitkeep placeholder, then: {dvc} add <data-path>")
    print(
        f"\nProfiles: {', '.join(options.profiles)} | Environment: {options.environment_manager} | "
        f"Tracking: {options.tracking} | {date.today().year}"
    )
    return 0
