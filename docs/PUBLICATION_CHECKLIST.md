# GitHub publication checklist

Repository files cover community standards, CI, security scanning, dependency updates, packaging,
and releases. The remaining controls are GitHub settings and must be enabled by a repository owner.

## Before changing visibility to public

- [ ] Review the complete Git history for credentials, tokens, private URLs, personal data, and
      datasets. Removing a current file does not remove it from history.
- [ ] Review all tracked and untracked files with `git status` and `git diff`.
- [ ] Confirm the MIT license and copyright holder are correct.
- [ ] Confirm all documentation links use the final repository owner/name.
- [ ] Run local linting, formatting, tests, build, and metadata validation.
- [ ] Generate and execute representative projects without relying on machine-local files.
- [ ] Create the initial changelog entry and semantic tag plan.

## Repository overview settings

- [ ] Set the description to: `Interactive generator for reproducible AI/ML experiment projects`.
- [ ] Add topics such as `python`, `machine-learning`, `artificial-intelligence`, `mlops`,
      `jupyter`, `uv`, `conda`, `dvc`, `mlflow`, and `project-template`.
- [ ] Set the website to the documentation or latest release when one is available.
- [ ] Enable Issues and, if desired, GitHub Discussions for usage questions and ideas.

## Branch and pull-request rules

- [ ] Protect `main` with a branch ruleset.
- [ ] Require pull requests before merging.
- [ ] Require the CI test matrix, package build, and CodeQL checks.
- [ ] Require branches to be up to date before merge.
- [ ] Block force pushes and branch deletion.
- [ ] Require conversation resolution.
- [ ] Prefer squash merges for a concise project history.
- [ ] Apply CODEOWNERS review requirements when additional maintainers join.

For a solo-maintained project, decide whether administrator bypass remains available for emergency
repairs; do not silently bypass failed required checks for normal changes.

## Security settings

- [ ] Enable the dependency graph.
- [ ] Enable Dependabot alerts, security updates, and version updates.
- [ ] Enable secret scanning and push protection.
- [ ] Enable private vulnerability reporting so `SECURITY.md` links work for external reporters.
- [ ] Confirm CodeQL uploads results to the Security tab.
- [ ] Review workflow token defaults and keep them read-only unless a job explicitly needs write.

## Community and maintenance

- [ ] Review **Insights → Community Standards** after the repository is public.
- [ ] Create labels used by templates: `bug`, `enhancement`, `dependencies`, `needs-triage`,
      `python`, and `github-actions`.
- [ ] Enable Discussions and update `SUPPORT.md` if Q&A should move there.
- [ ] Define a private conduct-reporting contact method on the maintainer profile.
- [ ] Triage issues and dependency pull requests on a documented cadence.

## Releases and distribution

- [ ] Push the first semantic tag only after CI succeeds on the release commit.
- [ ] Verify the release workflow attaches both wheel and source distributions.
- [ ] Enable immutable releases if they fit the maintenance process.
- [ ] Configure TestPyPI/PyPI trusted publishing before claiming PyPI installation support.
- [ ] Update installation documentation after the package is actually available from PyPI.
