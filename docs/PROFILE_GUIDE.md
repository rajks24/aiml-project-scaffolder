# Profile guide

Profiles are composable dependency sets exposed as standard Python optional dependencies. They let
one scaffold support lightweight classical ML, deep learning, GenAI, and Hugging Face workflows
without installing every framework into every environment.

## Selection guide

| Project type | Install profiles | Why |
|---|---|---|
| Classical ML | `general` | NumPy, pandas, and scikit-learn |
| Tabular ML | `general` + `tabular` | Adds Arrow, imbalance handling, and XGBoost |
| Image/deep learning | `deep-learning` | PyTorch, torchvision, Lightning, and NumPy |
| Deep learning with preprocessing/metrics | `general` + `deep-learning` | Adds pandas/scikit-learn preprocessing and metrics |
| Basic OpenAI/LLM application | `genai` | OpenAI client, Pydantic, and retry support |
| GenAI with CSV/data analysis | `general` + `genai` | Adds dataframe and classical analysis tooling |
| Hugging Face dataset/evaluation workflow | `general` + `genai` + `hf` | Adds datasets, Transformers, Evaluate, and Accelerate |
| RAG with embeddings/clustering/evaluation | `general` + `genai` | General ML covers vector analysis and evaluation baselines |

`tabular` intentionally contains only the packages beyond `general`. Install both. The
`deep-learning` profile can stand alone for image/deep-learning work. The `hf` profile is separate
from `genai` so an OpenAI-only application does not pull in the larger Hugging Face stack.

## Select profiles when scaffolding

Repeat `--profile` for each required profile:

```bash
uv run aiml-scaffold vision-lab \
  --profile general \
  --profile deep-learning

uv run aiml-scaffold llm-eval \
  --profile general \
  --profile genai \
  --profile hf
```

The interactive wizard accepts comma-separated profile numbers, such as `1,4,5`.

## Install or change profiles with uv

Generated profiles are `[project.optional-dependencies]` entries, so normal uv extra flags apply:

```bash
uv sync --extra general
uv sync --extra general --extra tabular
uv sync --extra general --extra deep-learning
uv sync --extra general --extra genai
uv sync --extra general --extra genai --extra hf
uv sync --extra general --extra deep-learning --extra genai
```

Add `--group notebooks` when Jupyter is needed. Commit the updated `uv.lock` after changing extras.

## Install or change profiles with Conda

For Conda projects, the generated `environment.yml` installs the package editable with selected
extras:

```yaml
- pip:
    - "-e .[general,genai,hf]"
```

To change profiles, edit that extra list and the `profile` value in `configs/experiment.toml`, then:

```bash
conda env update --file environment.yml --prune
```

The editable install ensures notebooks import the current `src/` code without manipulating
`sys.path`.

## Maintain the catalog

All package lower bounds live in
[`aiml_scaffolder/dependency_catalog.toml`](../aiml_scaffolder/dependency_catalog.toml):

- `dependencies.core`: runtime packages included in every project.
- `profiles.<name>.packages`: packages exposed by an optional profile.
- `tracking.<name>.packages`: tracking-backend dependencies.
- `data-versioning.<name>.packages`: data-versioning tools.
- `groups.dev` and `groups.notebooks`: local development dependency groups.

Use lower bounds in the catalog and exact lock files in generated projects. After editing the
catalog, run the test suite and generate representative uv, Conda, and `both` projects.

## GPU considerations

The catalog deliberately avoids hard-coding CUDA-specific PyTorch indexes. GPU installation varies
by operating system, accelerator, and driver. Generate the project first, then configure the
appropriate PyTorch source/index in its `pyproject.toml` and regenerate its lock file.
