# modeling.R
library(e1071)       # for SVM
library(randomForest)
library(caret)
library(keras)       # deep learning - needs keras & tensorflow backend installed
library(tensorflow)
library(pROC)

run_models <- function(X_train, y_train, X_test, y_test, outdir="artifacts") {
  dir.create(outdir, showWarnings = FALSE)
  results <- list()
  # Combine for caret training convenience
  train_df <- cbind(X_train, class = y_train)
  test_df <- cbind(X_test, class = y_test)
  # 1. SVM One-vs-All (OAA) - implement by training one classifier per class
  classes <- levels(as.factor(y_train))
  oaa_models <- list()
  preds_oaa <- matrix(NA, nrow = nrow(X_test), ncol = length(classes))
  for (i in seq_along(classes)) {
    cls <- classes[i]
    y_bin <- factor(ifelse(y_train == cls, cls, paste0("not_", cls)))
    df_train_bin <- cbind(X_train, y_bin = y_bin)
    model <- svm(y_bin ~ ., data = df_train_bin, probability = TRUE)
    oaa_models[[cls]] <- model
    probs <- attr(predict(model, X_test, probability = TRUE), "probabilities")[,cls]
    preds_oaa[,i] <- probs
  }
  # choose class with highest probability
  pred_oaa_class <- classes[apply(preds_oaa, 1, which.max)]
  results$svm_oaa <- list(pred = pred_oaa_class, models = oaa_models)
  cm_oaa <- confusionMatrix(factor(pred_oaa_class, levels=classes), factor(y_test, levels=classes))
  write.csv(as.data.frame(cm_oaa$table), file.path(outdir, "confusion_svm_oaa.csv"))
  capture.output(cm_oaa, file = file.path(outdir, "svm_oaa_report.txt"))
  results$svm_oaa$metrics <- cm_oaa
  # 2. SVM One-vs-One (OAO) - e1071's svm does OVO by default for multiclass. Train single multiclass SVM.
  model_oao <- svm(class ~ ., data = train_df, probability = TRUE)
  pred_oao <- predict(model_oao, X_test)
  cm_oao <- confusionMatrix(pred_oao, factor(y_test, levels=classes))
  write.csv(as.data.frame(cm_oao$table), file.path(outdir, "confusion_svm_oao.csv"))
  capture.output(cm_oao, file = file.path(outdir, "svm_oao_report.txt"))
  results$svm_oao <- list(model = model_oao, pred = pred_oao, metrics = cm_oao)
  # 3. Random Forest (ensemble)
  rf_model <- randomForest(class ~ ., data = train_df, ntree = 500)
  pred_rf <- predict(rf_model, X_test)
  cm_rf <- confusionMatrix(pred_rf, factor(y_test, levels=classes))
  write.csv(as.data.frame(cm_rf$table), file.path(outdir, "confusion_rf.csv"))
  capture.output(cm_rf, file = file.path(outdir, "rf_report.txt"))
  results$rf <- list(model = rf_model, pred = pred_rf, metrics = cm_rf)
  # 4. Deep Learning (Dense network) - using keras
  # Convert classes to integers and one-hot encode
  y_train_int <- as.integer(factor(y_train, levels=classes)) - 1
  y_test_int <- as.integer(factor(y_test, levels=classes)) - 1
  num_classes <- length(classes)
  x_train_mat <- as.matrix(X_train)
  x_test_mat <- as.matrix(X_test)
  y_train_oh <- to_categorical(y_train_int, num_classes = num_classes)
  y_test_oh <- to_categorical(y_test_int, num_classes = num_classes)
  model <- keras_model_sequential() %>%
    layer_dense(units = 128, activation = "relu", input_shape = ncol(x_train_mat)) %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = num_classes, activation = "softmax")
  model %>% compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = c("accuracy"))
  history <- model %>% fit(x_train_mat, y_train_oh, epochs = 30, batch_size = 32, validation_split = 0.2, verbose=0)
  scores <- model %>% evaluate(x_test_mat, y_test_oh, verbose=0)
  pred_dl_probs <- model %>% predict(x_test_mat)
  pred_dl <- classes[apply(pred_dl_probs, 1, which.max)]
  cm_dl <- confusionMatrix(factor(pred_dl, levels=classes), factor(y_test, levels=classes))
  capture.output(cm_dl, file = file.path(outdir, "dl_report.txt"))
  results$deep_learning <- list(model = model, history = history, pred = pred_dl, metrics = cm_dl, scores = scores)
  # Save models and results
  saveRDS(results, file = file.path(outdir, "model_results.rds"))
  return(results)
}
