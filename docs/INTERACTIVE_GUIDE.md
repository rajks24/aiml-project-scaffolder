# Interactive mode guide

Interactive mode asks for project metadata and technical choices, then prints environment-specific
next steps. Press Enter at any prompt to accept the displayed default.

## Start the wizard

From the scaffolder repository:

```bash
uv run aiml-scaffold
```

The compatibility command works as well:

```bash
python create_aiml_project.py
```

When run from the scaffolder source checkout, projects are created beside the repository. For
example:

```text
~/projects/aiml-project-scaffolder
~/projects/my-new-project
```

Use `--path` to choose another workspace:

```bash
uv run aiml-scaffold --path ~/experiments
```

## Understanding the prompts

```text
Project folder name [my-aiml-experiment]:
Project title [My Aiml Experiment]:
Author/team [your-user]:

Experiment profiles
  1. general (recommended default)
  2. tabular
  3. deep-learning
  4. genai
  5. hf
Choose one or more, comma-separated [1]:

Experiment tracking
  1. local (recommended)
  2. mlflow
Choose 1-2 [1]:

Data versioning
  1. none (recommended)
  2. dvc
Choose 1-2 [1]:

Environment manager
  1. uv (recommended)
  2. conda
  3. both
Choose 1-3 [1]:

Include GitHub Actions CI? [Y/n]:
```

Profile selections are comma-separated numbers. For example:

- `1` selects `general`.
- `1,2` selects `general + tabular`.
- `1,3` selects `general + deep-learning`.
- `1,4` selects `general + genai`.
- `1,4,5` selects `general + genai + hf`.

The numbers follow the displayed menu. Do not enter profile names at this prompt.

## Demo 1: classical machine learning with uv

Inputs are shown after each colon; blank responses accept defaults.

```text
$ uv run aiml-scaffold
Project folder name [my-aiml-experiment]: churn-baseline
Project title [Churn Baseline]: Customer Churn Baseline
Author/team [your-user]: Data Science Team
Experiment profiles ... [1]: 1
Experiment tracking ... [1]:
Data versioning ... [1]:
Environment manager ... [1]: 1
Include GitHub Actions CI? [Y/n]:
```

Then run:

```bash
cd ../churn-baseline
uv sync --extra general --group notebooks
uv run experiment --config configs/experiment.toml
uv run pytest
uv run jupyter lab
```

Equivalent non-interactive command:

```bash
uv run aiml-scaffold churn-baseline \
  --title "Customer Churn Baseline" \
  --author "Data Science Team" \
  --profile general \
  --environment uv
```

## Demo 2: tabular ML with MLflow, DVC, and Conda

```text
$ uv run aiml-scaffold
Project folder name [my-aiml-experiment]: fraud-detection
Project title [Fraud Detection]:
Author/team [your-user]: Risk Analytics
Experiment profiles ... [1]: 1,2
Experiment tracking ... [1]: 2
Data versioning ... [1]: 2
Environment manager ... [1]: 2
Include GitHub Actions CI? [Y/n]: y
```

Then run:

```bash
cd ../fraud-detection
conda env create --file environment.yml
conda activate fraud-detection
git init
dvc init
experiment --config configs/experiment.toml
pytest
mlflow ui
```

Before running `dvc add`, put the dataset in `data/raw/`, remove the directory's `.gitkeep`, and
track the actual path:

```bash
rm data/raw/.gitkeep
dvc add data/raw/transactions.csv
git add data/raw/transactions.csv.dvc data/raw/.gitignore
```

Equivalent non-interactive command:

```bash
uv run aiml-scaffold fraud-detection \
  --author "Risk Analytics" \
  --profile general \
  --profile tabular \
  --tracking mlflow \
  --data-versioning dvc \
  --environment conda
```

## Demo 3: Hugging Face evaluation with uv

Select `general + genai + hf` using `1,4,5`:

```text
$ uv run aiml-scaffold
Project folder name [my-aiml-experiment]: llm-evaluation
Project title [Llm Evaluation]: LLM Evaluation Lab
Author/team [your-user]: AI Platform
Experiment profiles ... [1]: 1,4,5
Experiment tracking ... [1]: 2
Data versioning ... [1]: 1
Environment manager ... [1]: 1
Include GitHub Actions CI? [Y/n]: y
```

Then run:

```bash
cd ../llm-evaluation
uv sync --extra general --extra genai --extra hf --group notebooks
uv run experiment --config configs/experiment.toml
uv run pytest
uv run jupyter lab
```

Equivalent non-interactive command:

```bash
uv run aiml-scaffold llm-evaluation \
  --title "LLM Evaluation Lab" \
  --author "AI Platform" \
  --profile general \
  --profile genai \
  --profile hf \
  --tracking mlflow \
  --environment uv
```

## Demo 4: support both uv and Conda

Choose environment option `3` when contributors use different managers:

```text
Project folder name [my-aiml-experiment]: vision-research
Experiment profiles ... [1]: 1,3
Experiment tracking ... [1]: 1
Data versioning ... [1]: 2
Environment manager ... [1]: 3
Include GitHub Actions CI? [Y/n]: y
```

This generates `environment.yml`, uv metadata, and separate uv and Conda CI jobs. Contributors must
use one environment at a time.

uv path:

```bash
uv sync --extra general --extra deep-learning --group notebooks
uv run pytest
```

Conda path:

```bash
conda env create --file environment.yml
conda activate vision-research
pytest
```

## Use interactive mode with preselected values

`--interactive` forces the wizard even when a project name is supplied. Command-line values become
the prompt defaults:

```bash
uv run aiml-scaffold demo-project \
  --interactive \
  --profile general \
  --profile genai \
  --environment both \
  --tracking mlflow
```

This is useful for demonstrations because users can review or adjust the preselected architecture.

## Preview without creating files

For a non-interactive preview:

```bash
uv run aiml-scaffold demo-project \
  --profile general \
  --profile genai \
  --environment both \
  --dry-run
```

The command lists every planned file and writes nothing. Interactive answers can also be combined
with `--dry-run` by adding the flag before starting the wizard.

## Existing destination behavior

The generator preserves existing files by default and reports how many were skipped. This allows
new template files to be added without overwriting project-specific work.

```bash
uv run aiml-scaffold existing-project --profile general
```

Use `--force` only when intentionally replacing generated files:

```bash
uv run aiml-scaffold existing-project --profile general --force
```

Review version-control changes before keeping a forced regeneration.

## Cancel or correct an answer

- Press `Ctrl+C` to cancel without continuing.
- Invalid menu input is rejected and the prompt is repeated.
- There is no back button. Cancel and restart if an earlier answer must change.
- Use `--dry-run` for complex non-interactive combinations before generating them.
