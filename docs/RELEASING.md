# Release guide

Releases follow [Semantic Versioning](https://semver.org/). The package version has one source of
truth: `aiml_scaffolder/__init__.py`.

## Choose the version

- Patch: backward-compatible fixes and dependency-bound corrections.
- Minor: backward-compatible features, profiles, integrations, or generated files.
- Major: incompatible CLI behavior, generated-project contracts, or supported-version changes.

## Prepare a release pull request

1. Confirm `main` is green and all intended work is merged.
2. Change `__version__` in `aiml_scaffolder/__init__.py`.
3. Move relevant `CHANGELOG.md` entries from `Unreleased` into a dated version section.
4. Update comparison links at the bottom of the changelog.
5. Update `CITATION.cff` version and release date.
6. Run:

   ```bash
   uv lock
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest
   uv build
   uvx twine check dist/*
   ```

7. Generate representative uv, Conda, DVC, MLflow, multi-profile, and Hugging Face projects when
   the release changes those areas.
8. Merge the release pull request after CI passes.

## Tag and publish the GitHub release

Create a signed semantic tag that exactly matches the package version:

```bash
git switch main
git pull --ff-only
git tag -s v2.1.0 -m "Release v2.1.0"
git push origin v2.1.0
```

The release workflow verifies the tag/version match, reruns quality checks, builds source and wheel
distributions, validates metadata, creates the GitHub release, and attaches both distributions.

If signed tags are not configured, use an annotated tag (`git tag -a`) while you set up signing.

## PyPI publishing

The repository does not publish to PyPI automatically yet. Configure a PyPI project and GitHub
trusted publisher first; then add an environment-protected publish job using OIDC. Do not add a
long-lived PyPI token unless trusted publishing is unavailable and its security tradeoff is accepted.

Test the first packaging release with TestPyPI before enabling production publication.
