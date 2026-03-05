#!/usr/bin/env python3
"""AIML Project Scaffolder

Creates a reproducible ML project structure with starter files:

- data/{raw,interim,processed}
- notebooks/ (placeholder notebooks)
- src/ (config, io, validation, cleaning, features, modeling, viz)
- scripts/ (build/validate dataset)
- reports/{figures}
- README.md, DATA_SOURCES.md, requirements.txt, LICENSE, .gitignore

Usage:
  python create_aiml_project.py my-project
  python create_aiml_project.py my-project --title "My Kaggle Project" --author "Rajesh Singh"
  python create_aiml_project.py my-project --path /some/folder
  python create_aiml_project.py my-project --force
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path


# -------------------------
# File templates
# -------------------------

GITIGNORE = """# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
.venv/
venv/
env/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# Data (keep structure, not raw data)
data/raw/*
data/interim/*
data/processed/*

# Reports
reports/figures/*

# OS
.DS_Store
Thumbs.db
"""

REQUIREMENTS_TXT = """pandas>=2.2
numpy>=1.26
matplotlib>=3.8
scikit-learn>=1.5
scipy>=1.13
jupyter>=1.0
"""

LICENSE_MIT = """MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

README_MD = """# {title}

A reproducible AI/ML project template (Kaggle + GitHub friendly) with a clean structure for:
- data intake and preprocessing
- validation + cleaning
- EDA
- feature engineering
- modeling (unsupervised + supervised)
- reporting and publication

## Project Layout

```text
{project_name}/
├─ data/
│  ├─ raw/           # original files (never edit)
│  ├─ interim/       # intermediate transforms
│  └─ processed/     # ML-ready datasets
├─ notebooks/
│  ├─ 01_data_intake.ipynb
│  ├─ 02_validation_and_cleaning.ipynb
│  ├─ 03_eda.ipynb
│  ├─ 04_feature_engineering.ipynb
│  ├─ 05_ml_unsupervised.ipynb
│  ├─ 06_ml_supervised.ipynb
│  └─ 07_story_notebook_kaggle.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ io_download.py
│  ├─ io_parse.py
│  ├─ validation.py
│  ├─ cleaning.py
│  ├─ features.py
│  ├─ modeling.py
│  └─ viz.py
├─ scripts/
│  ├─ build_dataset.py
│  └─ validate_dataset.py
├─ reports/
│  ├─ figures/
│  └─ final_report.md
├─ requirements.txt
├─ LICENSE
├─ .gitignore
├─ DATA_SOURCES.md
└─ README.md
```

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\\Scripts\\activate  # windows
pip install -r requirements.txt
```

### 2) Build and validate dataset
```bash
python scripts/build_dataset.py
python scripts/validate_dataset.py
```

## Notes
- Keep `data/raw/` unchanged.
- Put outputs into `data/processed/`.
- Keep notebooks story-driven; move reusable code to `src/`.

## Author
{author}
"""

DATA_SOURCES_MD = """# Data Sources

## Primary Source
- Dataset name:
- Provider:
- License:
- URL:

## Attribution Text (copy/paste)
> Data source: <Provider> — <Dataset name>, licensed under <license>. <URL>

## Notes
- Keep raw downloads in data/raw/ unchanged.
- Document transformations and derived features in notebooks and reports/final_report.md.
"""

FINAL_REPORT_MD = """# Final Report

## 1. Dataset Overview
- Source, license, coverage

## 2. Data Validation Summary
- Missingness, duplicates, schema checks

## 3. EDA Highlights
- Key charts and insights

## 4. Feature Engineering
- Derived variables and rationale

## 5. ML Results
### Unsupervised
- Clusters, anomalies

### Supervised (optional)
- Target definition, models, evaluation

## 6. Conclusions & Next Steps
- What worked, limitations, next improvements
"""

SRC_INIT = "# Package init\n"

SRC_CONFIG = """from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

# Example main output
OUT_DATASET = DATA_PROCESSED / "dataset.csv"
"""

SRC_VALIDATION = """import pandas as pd


def basic_schema_checks(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def summarize_missingness(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().sort_values(ascending=False)


def find_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    return df[df.duplicated(subset=subset, keep=False)].copy()
"""

SRC_CLEANING = """import pandas as pd

NULL_STRINGS = {"", "NA", "N/A", "null", "None", "-"}


def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].replace(list(NULL_STRINGS), pd.NA)
    return out


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
"""

SRC_FEATURES = """import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add derived features here
    return df.copy()
"""

SRC_MODELING = """from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


def train_baseline_classifier(X, y):
    model = LogisticRegression(max_iter=200)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    return model, scores
"""

SRC_VIZ = """import matplotlib.pyplot as plt


def save_histogram(series, out_path):
    plt.figure()
    series.dropna().plot(kind="hist")
    plt.title(getattr(series, "name", "value"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
"""

SCRIPT_BUILD = """from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED, OUT_DATASET


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # TODO: Replace with your data intake logic.
    raw_files = list(Path(DATA_RAW).glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(
            "No CSV found in data/raw/. Put a raw CSV there or update build logic."
        )

    df = pd.read_csv(raw_files[0])
    df.to_csv(OUT_DATASET, index=False)
    print(f"✅ Wrote dataset: {OUT_DATASET}")


if __name__ == "__main__":
    main()
"""

SCRIPT_VALIDATE = """import pandas as pd

from src.config import OUT_DATASET
from src.validation import basic_schema_checks, summarize_missingness


def main():
    df = pd.read_csv(OUT_DATASET)

    # TODO: Replace required_cols with your dataset schema
    required_cols = list(df.columns[: min(4, len(df.columns))])  # placeholder
    basic_schema_checks(df, required_cols)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("\nMissingness (top 10):")
    print(summarize_missingness(df).head(10))

    print("\n✅ Validation passed (basic).")


if __name__ == "__main__":
    main()
"""

NOTEBOOK_PLACEHOLDER = """{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {title}\\n",
    "\\n",
    "This notebook is a placeholder. Add your workflow steps here.\\n"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.11"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 5
}}
"""


@dataclass(frozen=True)
class ProjectMeta:
    project_name: str
    title: str
    author: str
    year: int


def write_file(path: Path, content: str, force: bool) -> None:
    """Write a file only if it doesn't exist, unless force=True."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    path.write_text(content, encoding="utf-8")


def create_notebook(path: Path, title: str, force: bool) -> None:
    content = NOTEBOOK_PLACEHOLDER.format(title=title)
    write_file(path, content, force=force)


def scaffold(meta: ProjectMeta, base_dir: Path, force: bool) -> Path:
    root = base_dir / meta.project_name

    # Create directories
    dirs = [
        root / "data" / "raw",
        root / "data" / "interim",
        root / "data" / "processed",
        root / "notebooks",
        root / "src",
        root / "scripts",
        root / "reports" / "figures",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Root files
    write_file(root / ".gitignore", GITIGNORE, force)
    write_file(root / "requirements.txt", REQUIREMENTS_TXT, force)
    write_file(root / "LICENSE", LICENSE_MIT.format(year=meta.year, author=meta.author), force)
    write_file(root / "DATA_SOURCES.md", DATA_SOURCES_MD, force)
    write_file(root / "reports" / "final_report.md", FINAL_REPORT_MD, force)
    write_file(
        root / "README.md",
        README_MD.format(title=meta.title, project_name=meta.project_name, author=meta.author),
        force,
    )

    # src package
    write_file(root / "src" / "__init__.py", SRC_INIT, force)
    write_file(root / "src" / "config.py", SRC_CONFIG, force)
    write_file(root / "src" / "validation.py", SRC_VALIDATION, force)
    write_file(root / "src" / "cleaning.py", SRC_CLEANING, force)
    write_file(root / "src" / "features.py", SRC_FEATURES, force)
    write_file(root / "src" / "modeling.py", SRC_MODELING, force)
    write_file(root / "src" / "viz.py", SRC_VIZ, force)
    write_file(root / "src" / "io_download.py", "# Put download/API logic here\n", force)
    write_file(root / "src" / "io_parse.py", "# Put file parsing logic here\n", force)

    # scripts
    write_file(root / "scripts" / "build_dataset.py", SCRIPT_BUILD, force)
    write_file(root / "scripts" / "validate_dataset.py", SCRIPT_VALIDATE, force)

    # Notebooks (placeholders)
    notebook_specs = [
        ("01_data_intake.ipynb", "01 - Data Intake"),
        ("02_validation_and_cleaning.ipynb", "02 - Validation & Cleaning"),
        ("03_eda.ipynb", "03 - Exploratory Data Analysis (EDA)"),
        ("04_feature_engineering.ipynb", "04 - Feature Engineering"),
        ("05_ml_unsupervised.ipynb", "05 - Unsupervised ML (Clustering/Anomalies)"),
        ("06_ml_supervised.ipynb", "06 - Supervised ML (Optional)"),
        ("07_story_notebook_kaggle.ipynb", "07 - Kaggle Story Notebook"),
    ]
    for fname, title in notebook_specs:
        create_notebook(root / "notebooks" / fname, title, force)

    return root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a reusable AIML project structure.")
    p.add_argument("project_name", help="Folder name for the project (e.g., ids-ml-project)")
    p.add_argument("--title", default=None, help="Human-friendly project title for README")
    p.add_argument("--author", default="Rajesh Singh", help="Author name")
    p.add_argument(
        "--path",
        "--base-dir",
        dest="base_dir",
        default=".",
        help="Directory where the project folder will be created (default: current directory).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    return p.parse_args()



def main() -> None:
    args = parse_args()

    project_name = args.project_name.strip()
    title = args.title or project_name.replace("-", " ").replace("_", " ").title()

    meta = ProjectMeta(
        project_name=project_name,
        title=title,
        author=args.author,
        year=date.today().year,
    )

    root = scaffold(meta, Path(args.base_dir).resolve(), force=args.force)

    print(f"✅ Project scaffold created at: {root}")
    print("Next steps:")
    print(f"  1) cd {root.name}")
    print("  2) python -m venv .venv")
    print("  3) source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows")
    print("  4) pip install -r requirements.txt")


if __name__ == "__main__":
    main()
