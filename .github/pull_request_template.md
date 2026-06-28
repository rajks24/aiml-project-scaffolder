## Summary

<!-- Explain the user need and the resulting behavior. -->

## Related issue

<!-- Use "Closes #123" when appropriate. -->

## Changes

-

## Verification

<!-- List exact commands and relevant generated-project combinations tested. -->

```text
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Checklist

- [ ] The change is focused and has no unrelated modifications.
- [ ] Tests cover code or generated-template behavior.
- [ ] User-visible behavior is documented and `CHANGELOG.md` is updated when appropriate.
- [ ] Dependency changes use `aiml_scaffolder/dependency_catalog.toml` and explain package weight,
      compatibility, maintenance, and license.
- [ ] Generated Python, TOML, YAML, uv, Conda, Jupyter, DVC, and CI behavior were considered.
- [ ] No secrets, credentials, private datasets, environments, or generated artifacts are included.
- [ ] I have read and followed `CONTRIBUTING.md` and the Code of Conduct.

## Screenshots or generated output

<!-- Include only when it materially helps reviewers. Remove sensitive data. -->
