# AIML Project Scaffolder

## Overview

This project provides a reproducible template for AI/ML workflows, suitable for Kaggle and GitHub. It helps you quickly set up:

- Data intake and preprocessing
- Validation and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Modeling (unsupervised and supervised)
- Reporting and publication

## What This Script Creates

When you run `create_aiml_project.py`, it generates the following structure:

```text
your_project_name/
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

## How to Use

1. **Create a new project folder:**
   ```bash
   python create_aiml_project.py my-project
   # Optional: add --title and --author
   python create_aiml_project.py my-project --title "My Kaggle Project" --author "Your Name"
   ```
2. **Navigate to your project:**
   ```bash
   cd my-project
   ```
3. **Set up your Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # mac/linux
   # .venv\Scripts\activate   # windows
   pip install -r requirements.txt
   ```
4. **Build and validate your dataset:**
   ```bash
   python scripts/build_dataset.py
   python scripts/validate_dataset.py
   ```

## Notes

- Keep `data/raw/` unchanged.
- Put outputs into `data/processed/`.
- Keep notebooks story-driven; move reusable code to `src/`.

## Author

Rajesh Singh
