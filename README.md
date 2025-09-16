# Arrhythmia CRISP-DM Project (R)

## Overview
This project implements a CRISP-DM style pipeline for multiclass arrhythmia classification using R. It includes:
- Missing value imputation (mice or median)
- Exploratory Data Analysis (EDA)
- Handling class imbalance with SMOTE
- Multiple modeling approaches:
  - SVM One-vs-All (OAA)
  - SVM One-vs-One (OAO) (multiclass SVM)
  - Random Forest ensemble
  - Deep Learning (Keras dense network)
- Evaluation metrics and artifacts export

## Files
- `main.R` - orchestrates the pipeline
- `data_acquisition.R` - load dataset
- `data_preparation.R` - imputation, scaling, SMOTE, train/test split
- `eda.R` - exploratory analysis and plots
- `modeling.R` - trains models, evaluates, and saves results
- `install_R_packages.R` - helper script to install required R packages
- `artifacts/` - (created after run) contains CSVs, plots, model objects, and reports
- `README.md` - this file

## How to run
1. Place `arrhythmia.csv` in the project root.
2. Open R (or RStudio) and run:
   ```r
   source("install_R_packages.R")   # optional, installs required packages
   Rscript main.R
   ```
3. Inspect `artifacts/` for outputs: confusion matrices, model reports, saved models, and pipeline_results.rds

## Notes & Recommendations
- Keras/TensorFlow installation in R requires system-level setup; if it's difficult, you can skip deep learning or run DL in Python.
- The script uses SMOTE from the `DMwR` package. If you have issues, consider `smotefamily` or `themis` in tidymodels.
- SVM OAA is implemented manually by training binary SVMs; OAO uses e1071's multiclass behavior (which is OVO by default).
- Tune hyperparameters using caret's train() and cross-validation for production-grade performance.
- Documentation maps each script to CRISP-DM phases: Business understanding (project goal), Data understanding (EDA), Data preparation (imputation, SMOTE), Modeling, Evaluation, Deployment (model objects saved).

If you want, I can:
- Run this pipeline here (if you upload `arrhythmia.csv`), produce artifacts, and return a zip that includes `artifacts/`.
- Extend models: add CNN/LSTM architectures (requires time-series shape or raw signals).
- Add hyperparameter tuning and cross-validation to optimize metrics like F1 macro, sensitivity per class.
