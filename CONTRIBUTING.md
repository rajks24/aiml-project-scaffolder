# Contributing

Contributions are welcome: bug fixes, new profiles, dependency updates, documentation, tests, and
improvements to generated project practices. Please keep changes focused and explain the user need
they address.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before opening an issue

1. Search existing issues and pull requests.
2. Check the [documentation](docs/) and run the latest `main` branch when practical.
3. Use the appropriate issue form and include enough information to reproduce or evaluate the
   request.
4. Report security concerns privately according to [SECURITY.md](SECURITY.md).

Questions and setup help can use a regular issue until GitHub Discussions is enabled.

## Development setup

Requirements:

- Git
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/rajks24/aiml-project-scaffolder.git
cd aiml-project-scaffolder
uv sync
uv run pre-commit install
uv run pytest
```

Create a feature branch from current `main`:

```bash
git switch -c feature/short-description
```

## Quality checks

Run these before submitting a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

If formatting fails:

```bash
uv run ruff format .
```

## Changing dependencies or profiles

The central source is
[`aiml_scaffolder/dependency_catalog.toml`](aiml_scaffolder/dependency_catalog.toml). Do not copy
package versions directly into unrelated templates.

When adding or updating a package:

1. Explain which workflow requires it and why an existing profile is insufficient.
2. Put universal packages in `dependencies.core`; otherwise use the narrowest profile or
   integration section.
3. Consider installation size, supported Python/platform versions, license, security history, and
   whether the package is actively maintained.
4. Keep broad lower bounds in the catalog; generated projects create exact environment locks.
5. Update the profile guide and changelog.
6. Test relevant uv and Conda profile combinations.

See [Profile Guide](docs/PROFILE_GUIDE.md) for the intended profile boundaries.

## Changing templates

Templates affect future generated projects. Preserve these guarantees:

- `--dry-run` writes nothing.
- Existing files are not overwritten without `--force`.
- Generated Python, TOML, and YAML remain valid.
- uv, Conda, and `both` modes stay internally consistent.
- Generated projects pass their own Ruff and pytest checks.
- DVC metadata remains Git-trackable while local DVC state stays ignored.

Add or update tests in `tests/test_scaffolder.py`. For material template changes, generate a
temporary project and execute its documented setup, test, and experiment commands.

## Documentation changes

Documentation-only pull requests are welcome. Keep commands executable, avoid machine-specific
paths except when explicitly labeled as examples, and update both the main README and focused guide
when the same behavior is described in both places.

## Pull requests

- Keep one logical change per pull request.
- Describe motivation and user-visible behavior, not only implementation details.
- Link related issues using `Closes #123` when applicable.
- Include tests for code changes and documentation for user-visible changes.
- Confirm that no secrets, local datasets, generated environments, or unrelated files are present.
- Allow maintainers to edit the branch when contributing from a fork.

Maintainers may ask for changes to keep profile scope, dependency weight, compatibility, or
generated-project complexity under control.

## Commit and release policy

Clear, imperative commit subjects are preferred, for example `Add Hugging Face profile validation`.
Maintainers use semantic versions and maintain [CHANGELOG.md](CHANGELOG.md). Contributors should add
user-visible changes under `Unreleased`; maintainers finalize versions during release.
