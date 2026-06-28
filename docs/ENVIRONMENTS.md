# Environment and Jupyter guide

The scaffolder supports `uv`, `conda`, and `both`. All modes use the generated `pyproject.toml` as
the project metadata and runtime dependency authority.

## uv

uv creates an isolated `.venv`, installs the project, and locks all dependency resolutions:

```bash
uv sync --extra general --group notebooks
uv run experiment --config configs/experiment.toml
uv run pytest
```

Register its Jupyter kernel:

```bash
uv run python -m ipykernel install --user \
  --name my-project-uv \
  --display-name "Python (My Project - uv)"
uv run jupyter lab
```

## Conda

Conda creates the Python environment; the `pip` section installs the generated project editable and
reads its selected optional dependencies from `pyproject.toml`:

```bash
conda env create --file environment.yml
conda activate my-project
experiment --config configs/experiment.toml
pytest
```

Register its kernel:

```bash
python -m ipykernel install --user \
  --name my-project-conda \
  --display-name "Python (My Project - Conda)"
jupyter lab
```

Update an existing environment with `conda env update --file environment.yml --prune`.

## Both

`both` generates uv and Conda instructions plus separate CI jobs. Use one environment per shell
session. Do not activate Conda and `.venv` simultaneously; doing so makes Python, console scripts,
and Jupyter kernels ambiguous.

Check the active interpreter when diagnosing an environment issue:

```bash
python -c "import sys; print(sys.executable)"
python -c "import your_package; print(your_package.__file__)"
jupyter kernelspec list
```

## CI behavior

- `uv`: GitHub Actions installs uv, syncs selected extras and notebooks, then runs quality checks.
- `conda`: GitHub Actions creates `environment.yml` using setup-miniconda and runs inside the
  activated environment.
- `both`: both jobs run, catching dependency or packaging behavior that differs by manager.

## Common problems

### Notebook cannot import the package

Select the registered project kernel, confirm the interpreter path, and ensure the editable project
installation completed. Avoid adding ad hoc `sys.path` changes to notebooks.

### Conda and uv resolve different versions

Both consume the same lower bounds but use separate solvers. Commit `uv.lock`; for strict Conda
reproduction, add a platform-specific Conda lock/export to the generated project.

### Conda environment already exists

Use `conda env update --file environment.yml --prune`, or remove and recreate the environment if a
clean solve is required.

### Wrong Jupyter kernel

Compare `sys.executable` inside the notebook with the expected `.venv` or Conda environment path.
Remove stale registrations with `jupyter kernelspec uninstall <kernel-name>`.
