# data_preparation.R
library(mice)        # for multiple imputation
library(caret)       # for preprocessing, train/test split
library(DMwR)        # for SMOTE
library(dplyr)

prepare_data <- function(df, outdir="artifacts", impute_method="mice", smote_k=5) {
  dir.create(outdir, showWarnings = FALSE)
  # Remove columns with all NA or near-zero variance
  df <- df %>% select_if(~!all(is.na(.)))
  nzv <- nearZeroVar(df %>% select(-class), saveMetrics=TRUE)
  if (any(nzv$nzv)) {
    df <- df %>% select(-names(nzv[nzv$nzv,]))
  }
  # Split into features and target
  y <- df$class
  X <- df %>% select(-class)
  # Imputation
  if (impute_method == "mice") {
    # mice: numeric and categorical handling
    imp <- mice(X, m=1, maxit=5, method='pmm', seed=42, printFlag=FALSE)
    X_imp <- complete(imp, 1)
  } else if (impute_method == "median") {
    pre <- preProcess(X, method = c("medianImpute"))
    X_imp <- predict(pre, X)
  } else {
    stop("Unknown imputation method")
  }
  # Encoding: ensure factors handled; caret will one-hot encode if needed
  # Create train/test split stratified
  set.seed(42)
  trainIndex <- createDataPartition(y, p = .8, list = FALSE, times = 1)
  X_train <- X_imp[trainIndex, , drop=FALSE]; X_test <- X_imp[-trainIndex, , drop=FALSE]
  y_train <- y[trainIndex]; y_test <- y[-trainIndex]
  # Address imbalance with SMOTE on training set
  # Combine for SMOTE: SMOTE in DMwR expects a formula interface with class as factor
  train_df <- cbind(X_train, class = y_train)
  # Convert character columns to factors
  train_df[] <- lapply(train_df, function(col) if(is.character(col)) as.factor(col) else col)
  # Apply SMOTE (only if minority classes exist)
  try({
    train_smote <- SMOTE(class ~ ., train_df, perc.over = 200, k = smote_k, perc.under = 150)
    X_train_smote <- train_smote %>% select(-class)
    y_train_smote <- train_smote$class
  }, silent = TRUE)
  if (!exists("train_smote")) {
    X_train_smote <- X_train; y_train_smote <- y_train
  }
  # Scale numeric features
  num_cols <- sapply(X_train_smote, is.numeric)
  pre_scaler <- preProcess(X_train_smote[,num_cols], method = c("center", "scale"))
  X_train_smote[,num_cols] <- predict(pre_scaler, X_train_smote[,num_cols])
  X_test[,num_cols] <- predict(pre_scaler, X_test[,num_cols])
  # Save artifacts
  write.csv(X_train_smote, file.path(outdir, "X_train_smote.csv"), row.names = FALSE)
  write.csv(X_test, file.path(outdir, "X_test.csv"), row.names = FALSE)
  write.csv(data.frame(class = y_train_smote), file.path(outdir, "y_train_smote.csv"), row.names = FALSE)
  write.csv(data.frame(class = y_test), file.path(outdir, "y_test.csv"), row.names = FALSE)
  return(list(X_train = X_train_smote, y_train = y_train_smote, X_test = X_test, y_test = y_test,
              imputer = if (impute_method=="mice") "mice" else "median", scaler = pre_scaler))
}
