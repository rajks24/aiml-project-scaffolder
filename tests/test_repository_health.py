from __future__ import annotations

import re
from pathlib import Path

import yaml

from aiml_scaffolder import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_release_metadata_versions_match() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert citation["version"] == __version__
    assert f"## [{__version__}]" in changelog


def test_community_health_files_exist() -> None:
    required = (
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "SUPPORT.md",
        ".github/pull_request_template.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
    )
    assert all((ROOT / path).is_file() for path in required)


def test_repository_yaml_is_valid() -> None:
    yaml_files = [*ROOT.glob(".github/**/*.yml"), ROOT / "CITATION.cff"]
    for path in yaml_files:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None, path


def test_workflow_actions_are_pinned_to_commit_shas() -> None:
    workflows = "\n".join(
        path.read_text(encoding="utf-8") for path in ROOT.glob(".github/workflows/*.yml")
    )
    references = re.findall(r"^\s*uses:\s*([^\s#]+)", workflows, flags=re.MULTILINE)
    assert references
    for reference in references:
        assert re.search(r"@[0-9a-f]{40}$", reference), reference


def test_local_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for document in ROOT.glob("**/*.md"):
        if any(
            part.startswith(".") or part in {"dist"} for part in document.parts[len(ROOT.parts) :]
        ):
            continue
        for target in link_pattern.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative_target = target.split("#", 1)[0]
            assert (document.parent / relative_target).exists(), f"{document}: {target}"
