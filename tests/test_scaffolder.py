from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
import yaml

from aiml_scaffolder.catalog import DEPENDENCIES
from aiml_scaffolder.cli import _choose_many, build_parser, default_base_dir, main
from aiml_scaffolder.generator import scaffold
from aiml_scaffolder.models import ProjectOptions
from aiml_scaffolder.templates import project_files


def options(tmp_path: Path, **overrides: object) -> ProjectOptions:
    values = {
        "project_name": "credit-risk-lab",
        "title": "Credit Risk Lab",
        "author": "Test Team",
        "base_dir": tmp_path,
    }
    values.update(overrides)
    return ProjectOptions(**values)  # type: ignore[arg-type]


def test_scaffold_creates_runnable_project_shape(tmp_path: Path) -> None:
    result = scaffold(options(tmp_path))
    root = result.root
    assert (root / "pyproject.toml").is_file()
    assert (root / "src/credit_risk_lab/experiment.py").is_file()
    assert (root / ".github/workflows/ci.yml").is_file()
    assert len(result.created) >= 20


def test_generated_gitignore_excludes_outputs_but_keeps_placeholders(tmp_path: Path) -> None:
    generated = project_files(options(tmp_path))[".gitignore"]
    assert ".env.*" in generated
    assert "data/raw/**" in generated
    assert "artifacts/runs/**" in generated
    assert "mlruns/" in generated
    assert "wandb/" in generated
    assert "!data/raw/.gitkeep" in generated


def test_dvc_metadata_is_trackable_but_local_state_is_ignored(tmp_path: Path) -> None:
    generated = project_files(options(tmp_path, data_versioning="dvc"))
    gitignore = generated[".gitignore"]
    assert "data/raw/**" not in gitignore
    assert ".dvc/cache/" in gitignore
    assert ".dvc/tmp/" in gitignore
    assert ".dvc/config.local" in gitignore
    assert "*.dvc" not in gitignore.splitlines()
    assert ".dvcignore" in generated
    assert "dvc.yaml" in generated


def test_existing_files_are_preserved_without_force(tmp_path: Path) -> None:
    scaffold(options(tmp_path))
    readme = tmp_path / "credit-risk-lab/README.md"
    readme.write_text("mine\n", encoding="utf-8")
    result = scaffold(options(tmp_path, title="Changed"))
    assert readme.read_text(encoding="utf-8") == "mine\n"
    assert Path("README.md") in result.skipped


def test_profile_integrations_are_selected(tmp_path: Path) -> None:
    scaffold(options(tmp_path, profiles=("genai", "hf"), tracking="mlflow", data_versioning="dvc"))
    pyproject = (tmp_path / "credit-risk-lab/pyproject.toml").read_text(encoding="utf-8")
    assert "openai>=1.75" in pyproject
    assert "transformers>=4.51" in pyproject
    assert "mlflow>=3.0" in pyproject
    assert "dvc>=3.59" in pyproject


def test_dependency_catalog_drives_generated_packages(tmp_path: Path) -> None:
    generated = project_files(options(tmp_path, profiles=("general",), tracking="mlflow"))
    pyproject = generated["pyproject.toml"]
    expected = (
        *DEPENDENCIES.core,
        *DEPENDENCIES.profiles["general"],
        *DEPENDENCIES.tracking["mlflow"],
        *DEPENDENCIES.groups["dev"],
    )
    assert all(package in pyproject for package in expected)


@pytest.mark.parametrize("profile", ["general", "tabular", "deep-learning", "genai", "hf"])
@pytest.mark.parametrize("tracking", ["local", "mlflow"])
@pytest.mark.parametrize("environment_manager", ["uv", "conda", "both"])
def test_every_profile_generates_valid_python_and_toml(
    tmp_path: Path, profile: str, tracking: str, environment_manager: str
) -> None:
    generated = project_files(
        options(
            tmp_path,
            profiles=(profile,),
            tracking=tracking,
            environment_manager=environment_manager,
            title='A "quoted" title',
        )
    )
    for name, content in generated.items():
        if name.endswith(".py"):
            ast.parse(content, filename=name)
    tomllib.loads(generated["pyproject.toml"])
    tomllib.loads(generated["configs/experiment.toml"])
    yaml.safe_load(generated[".github/workflows/ci.yml"])
    if environment_manager in {"conda", "both"}:
        yaml.safe_load(generated["environment.yml"])


@pytest.mark.parametrize("environment_manager", ["uv", "conda", "both"])
def test_environment_manager_generates_matching_guidance_and_ci(
    tmp_path: Path, environment_manager: str
) -> None:
    generated = project_files(options(tmp_path, environment_manager=environment_manager))
    readme = generated["README.md"]
    ci = generated[".github/workflows/ci.yml"]
    config = tomllib.loads(generated["configs/experiment.toml"])
    assert config["project"]["environment_manager"] == environment_manager

    if environment_manager == "uv":
        assert "environment.yml" not in generated
        assert "astral-sh/setup-uv" in ci
        assert "uv run jupyter lab" in readme
    elif environment_manager == "conda":
        assert '"-e .[general]"' in generated["environment.yml"]
        assert "conda-incubator/setup-miniconda" in ci
        assert "astral-sh/setup-uv" not in ci
        assert "conda activate credit-risk-lab" in readme
    else:
        assert "environment.yml" in generated
        assert "astral-sh/setup-uv" in ci
        assert "conda-incubator/setup-miniconda" in ci
        assert "do not activate Conda and `.venv` together" in readme


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = scaffold(options(tmp_path), dry_run=True)
    assert result.created
    assert not result.root.exists()


def test_cli_noninteractive(tmp_path: Path) -> None:
    status = main(["demo", "--path", str(tmp_path), "--environment", "conda", "--no-ci"])
    assert status == 0
    assert not (tmp_path / "demo/.github").exists()
    assert (tmp_path / "demo/environment.yml").is_file()


def test_cli_accepts_repeated_profiles(tmp_path: Path) -> None:
    status = main(
        [
            "combo",
            "--path",
            str(tmp_path),
            "--profile",
            "general",
            "--profile",
            "genai",
            "--profile",
            "hf",
            "--no-ci",
        ]
    )
    assert status == 0
    config = tomllib.loads((tmp_path / "combo/configs/experiment.toml").read_text())
    assert config["project"]["profile"] == "general,genai,hf"


def test_interactive_profile_selection_accepts_comma_separated_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "1,4,5")
    assert _choose_many(
        "Profiles", ("general", "tabular", "deep-learning", "genai", "hf"), ("general",)
    ) == ("general", "genai", "hf")


def test_default_destination_is_next_to_source_repository() -> None:
    repository = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(["demo"])
    assert Path(args.base_dir) == repository.parent


def test_default_destination_falls_back_to_working_directory(tmp_path: Path) -> None:
    installed_module = tmp_path / "site-packages/aiml_scaffolder/cli.py"
    assert default_base_dir(installed_module, tmp_path) == tmp_path


def test_invalid_environment_manager_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="environment manager"):
        options(tmp_path, environment_manager="unknown")


def test_profile_combinations_become_uv_extras_and_conda_editable_extras(
    tmp_path: Path,
) -> None:
    profiles = ("general", "deep-learning", "genai")
    generated = project_files(options(tmp_path, profiles=profiles, environment_manager="both"))
    assert "uv sync --extra general --extra deep-learning --extra genai" in generated["README.md"]
    assert '"-e .[general,deep-learning,genai]"' in generated["environment.yml"]
    assert 'profile = "general,deep-learning,genai"' in generated["configs/experiment.toml"]


def test_invalid_or_empty_profiles_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one profile"):
        options(tmp_path, profiles=())
    with pytest.raises(ValueError, match="Unknown profiles"):
        options(tmp_path, profiles=("unknown",))


@pytest.mark.parametrize("name", ["../escape", "nested/project", "bad:name", " "])
def test_unsafe_project_names_are_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        options(tmp_path, project_name=name)
