# Architecture and maintenance

## Generation flow

```text
CLI arguments / interactive wizard
                |
                v
          ProjectOptions
                |
                v
 dependency catalog + templates
                |
                v
 safe filesystem generator
                |
                v
 generated package, config, CI, tests, data layout, and documentation
```

## Source modules

- `aiml_scaffolder/cli.py`: argument parsing, guided prompts, destination selection, and next steps.
- `aiml_scaffolder/models.py`: validated immutable project options.
- `aiml_scaffolder/catalog.py`: typed loading of the central TOML dependency catalog.
- `aiml_scaffolder/templates.py`: manager/profile-aware project content.
- `aiml_scaffolder/generator.py`: non-destructive filesystem writes and dry-run reporting.
- `create_aiml_project.py`: compatibility entry point.

Existing files are preserved unless `--force` is explicit. Use `--dry-run` before introducing a
new template or changing many generated files.

## Dependency ownership

The catalog owns package names and lower bounds. Generated `pyproject.toml` files expose profile
packages as optional dependencies. Selected extras are installed by uv, Conda's editable pip
install, and CI. This prevents three separate dependency lists from drifting.

## Data and DVC ownership

Without DVC, local data stages and generated models are ignored while `.gitkeep` files preserve the
directory structure. With DVC selected, data paths are not pre-ignored because `dvc add` must see
them. DVC then adds the large data path to `.gitignore` and creates Git-trackable metadata.

Ignore local DVC state:

- `.dvc/cache/`
- `.dvc/tmp/`
- `.dvc/config.local`

Commit reproducibility metadata:

- `.dvc/config`
- `.dvcignore`
- `*.dvc`
- `dvc.yaml`
- `dvc.lock`

## Change checklist

1. Update the dependency catalog or focused template helper.
2. Add a unit test for the new option or generated file.
3. Run Ruff formatting/linting and pytest.
4. Generate uv, Conda, and `both` examples.
5. Parse generated TOML/YAML and compile generated Python.
6. For dependency changes, run an actual environment sync and generated experiment.
7. Update the relevant documentation guide.
