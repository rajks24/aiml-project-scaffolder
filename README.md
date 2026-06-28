# AIML Project Scaffolder

[![CI](https://github.com/rajks24/aiml-project-scaffolder/actions/workflows/ci.yml/badge.svg)](https://github.com/rajks24/aiml-project-scaffolder/actions/workflows/ci.yml)
[![CodeQL](https://github.com/rajks24/aiml-project-scaffolder/actions/workflows/codeql.yml/badge.svg)](https://github.com/rajks24/aiml-project-scaffolder/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A dependency-free, interactive Python CLI that creates reproducible AI/ML experiment projects.
It generates a tested `src/` package, typed TOML configuration, run manifests, quality tooling,
and focused dependencies for the selected experiment type.

The project is in active beta development. Interfaces are tested, but generated defaults will evolve
as Python and AI/ML tooling changes.

## Install and run

Run the latest GitHub version without cloning:

```bash
uvx --from git+https://github.com/rajks24/aiml-project-scaffolder.git aiml-scaffold
```

Or clone the repository for development or repeat use:

```bash
git clone https://github.com/rajks24/aiml-project-scaffolder.git
cd aiml-project-scaffolder
uv sync
uv run aiml-scaffold
```

The package is not yet published to PyPI. GitHub releases contain validated wheel and source
distributions; the installation instructions will be updated when PyPI publishing is enabled.

## Quick start

Run the guided wizard:

```bash
uv run aiml-scaffold
```

See the [interactive mode guide](docs/INTERACTIVE_GUIDE.md) for complete demonstrations, prompt
examples, multi-profile selection, dry runs, and equivalent scripted commands.

When run from this source checkout, the default destination is the parent of this repository. For
example, running from `~/projects/aiml-project-scaffolder` creates projects under `~/projects`. Use
`--path` to choose a different workspace.

Or generate a project non-interactively:

```bash
uv run aiml-scaffold fraud-lab \
  --title "Fraud Detection Lab" \
  --author "Data Science Team" \
  --profile general \
  --profile tabular \
  --tracking mlflow \
  --data-versioning dvc \
  --environment both
```

The original command remains supported:

```bash
python create_aiml_project.py fraud-lab
```

Preview every file without writing:

```bash
uv run aiml-scaffold demo --dry-run
```

## Choices

- Profiles: `general`, `tabular`, `deep-learning`, `genai`, or `hf`.
- Tracking: portable local JSON manifests or MLflow.
- Data versioning: plain staged folders or DVC.
- Environment manager: `uv`, `conda`, or `both`.
- CI: GitHub Actions is included by default; use `--no-ci` to omit it.
- Python: 3.12 by default; choose 3.11 or newer with `--python`.

Profiles are repeatable and composable. The wizard accepts comma-separated selections.

| Project type | Recommended profiles |
|---|---|
| Classical ML | `general` |
| Tabular ML | `general` + `tabular` |
| Image/deep learning | `deep-learning` |
| Deep learning with preprocessing/metrics | `general` + `deep-learning` |
| Basic OpenAI/LLM application | `genai` |
| GenAI with CSV/data analysis | `general` + `genai` |
| Hugging Face dataset/evaluation | `general` + `genai` + `hf` |
| RAG with embeddings/clustering/evaluation | `general` + `genai` |

Generated projects expose profiles as optional dependencies:

```bash
uv sync --extra general
uv sync --extra general --extra deep-learning
uv sync --extra general --extra genai
uv sync --extra general --extra genai --extra hf
```

See the [profile guide](docs/PROFILE_GUIDE.md) for package ownership, Conda equivalents, changing
profiles later, and GPU considerations.

## Dependency maintenance

All generated dependency versions are managed in
`aiml_scaffolder/dependency_catalog.toml`. Add universal runtime packages to `dependencies.core`,
or update the relevant profile, tracking, data-versioning, or development group. Both the CLI
choices and generated `pyproject.toml` files read from this catalog, so dependency policy remains
in one place.

When DVC is selected, Git ignores `.dvc/cache/`, `.dvc/tmp/`, and `.dvc/config.local`. Commit
`.dvc/config`, `.dvcignore`, `*.dvc`, `dvc.yaml`, and `dvc.lock`; these are the reproducibility
metadata that points to data stored outside Git.

Every generated project includes uv dependency groups, Ruff, pytest, pre-commit, a valid package
entry point, configuration validation, data-source documentation, secret-safe defaults, and a
starter experiment that runs before any dataset is available.

## Generated workflows

uv projects use an isolated `.venv`:

```bash
cd fraud-lab
uv sync --extra general --extra tabular --group notebooks
uv run experiment --config configs/experiment.toml
uv run pytest
uv run ruff check .
```

Conda projects receive an `environment.yml` with the project installed editable:

```bash
cd fraud-lab
conda env create --file environment.yml
conda activate fraud-lab
experiment --config configs/experiment.toml
pytest
jupyter lab
```

Both workflows include `ipykernel` guidance so the correct environment can be selected from
Jupyter. Heavy ML and GenAI packages remain limited to the selected profiles.

## CLI reference

```text
aiml-scaffold [project-name]
  --profile PROFILE             Repeat to compose profiles
  --environment uv|conda|both
  --tracking local|mlflow
  --data-versioning none|dvc
  --python 3.12
  --path /workspace
  --[no-]ci
  --interactive
  --dry-run
  --force
```

Omitting the project name opens the guided wizard. `--dry-run` lists planned files without writing.
Existing files are preserved unless `--force` is supplied.

## Documentation

- [Interactive mode guide](docs/INTERACTIVE_GUIDE.md): complete wizard demonstrations and examples.
- [Profile guide](docs/PROFILE_GUIDE.md): combinations, extras, catalog maintenance, and GPU notes.
- [Environment and Jupyter guide](docs/ENVIRONMENTS.md): uv, Conda, kernels, CI, and troubleshooting.
- [Architecture](docs/ARCHITECTURE.md): generation flow, module ownership, DVC, and change checklist.

## Develop the scaffolder

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The implementation is split across input validation, safe filesystem generation, CLI interaction,
and templates under `aiml_scaffolder/`. Existing generated files are preserved unless `--force` is
explicitly supplied.

## Contributing and project policies

Contributions and focused package proposals are welcome. Start with
[CONTRIBUTING.md](CONTRIBUTING.md), use the GitHub issue forms, and include tests plus documentation
for user-visible behavior.

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Support](SUPPORT.md)
- [Changelog](CHANGELOG.md)
- [Release Guide](docs/RELEASING.md)
- [GitHub Publication Checklist](docs/PUBLICATION_CHECKLIST.md)

This project is available under the [MIT License](LICENSE).
