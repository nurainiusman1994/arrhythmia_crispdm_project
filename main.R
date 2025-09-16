# CRISP-DM Pipeline for Arrhythmia Multiclass Classification (R)
# This script orchestrates dataset loading, missing value treatment, EDA, modeling (SVM OAA/OAO, Random Forest ensemble, Deep Learning - Keras),
# handling class imbalance (SMOTE), evaluation, and artifact export.
#
# NOTE: Put the arrhythmia dataset CSV in the same folder and name it `arrhythmia.csv`.
# If you prefer to download using Python/kagglehub, do so and place the CSV here.
#
# Run: Rscript main.R

library(here)

# Source pipeline steps
source(here("data_acquisition.R"))
source(here("data_preparation.R"))
source(here("eda.R"))
source(here("modeling.R"))

# Orchestrate
outdir <- here("artifacts")
dir.create(outdir, showWarnings = FALSE)
set.seed(42)

# 1. Data acquisition
df <- load_data("arrhythmia.csv")

# 2. Data understanding & EDA
eda_results <- eda_report(df, outdir)

# 3. Data preparation: imputation, encoding, imbalance handling
prep <- prepare_data(df, outdir, impute_method = "mice", smote_k = 5)

# 4. Modeling & evaluation
model_results <- run_models(prep$X_train, prep$y_train, prep$X_test, prep$y_test, outdir)

# 5. Save summary
saveRDS(list(eda = eda_results, prep = prep, models = model_results), file = file.path(outdir, "pipeline_results.rds"))
cat("Pipeline complete. Artifacts in:", outdir, "\\n")
