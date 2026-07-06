# Clear everything
rm(list = ls())
gc()

# Load libraries
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(e1071)

set.seed(42)

# Read data
data <- read.csv("C:\\Users\\Dell\\Downloads\\cleaned_liver_data.csv")

# Clean column names
names(data) <- gsub(" ", "_", names(data))
names(data) <- gsub("\\.", "_", names(data))
names(data)[names(data) == "A_G_RATIO"] <- "AG_RATIO"

# Create target with VALID R NAMES (must start with a letter)
# Using "Healthy" and "Diseased" instead of 0 and 1
data$target <- ifelse(data$SELECTOR == 1, "Healthy", "Diseased")
data$target <- factor(data$target, levels = c("Healthy", "Diseased"))

# Remove original SELECTOR
data$SELECTOR <- NULL

# Check data
cat("Dataset dimensions:", dim(data), "\n")
cat("Target distribution:\n")
print(table(data$target))

# Split data
set.seed(42)
train_index <- createDataPartition(data$target, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

cat("\nTraining set size:", nrow(train_data))
cat("\nTest set size:", nrow(test_data))

# Separate features and target
X_train <- train_data[, !names(train_data) %in% c("target")]
y_train <- train_data$target
X_test <- test_data[, !names(test_data) %in% c("target")]
y_test <- test_data$target

# Scale features (important for Logistic Regression and SVM)
preprocess_params <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preprocess_params, X_train)
X_test_scaled <- predict(preprocess_params, X_test)

# Create data frames with target for caret
train_scaled <- cbind(X_train_scaled, target = y_train)
test_scaled <- cbind(X_test_scaled, target = y_test)

# Verify target levels are valid
cat("\nTarget levels in training data:\n")
print(levels(train_scaled$target))

# ============================================
# CROSS-VALIDATION SETUP
# ============================================
cv_control <- trainControl(method = "cv",
                           number = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           savePredictions = TRUE,
                           verboseIter = FALSE)

# ============================================
# 1. LOGISTIC REGRESSION
# ============================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("LOGISTIC REGRESSION\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

log_model <- train(target ~ .,
                   data = train_scaled,
                   method = "glm",
                   family = "binomial",
                   trControl = cv_control,
                   metric = "ROC")

print(log_model)

# Predictions
log_pred <- predict(log_model, newdata = test_scaled)
log_prob <- predict(log_model, newdata = test_scaled, type = "prob")
summary(log_pred)
summary(log_prob)

# Performance
log_cm <- confusionMatrix(log_pred, y_test)
log_roc <- roc(y_test, log_prob$Diseased)

cat("\n=== Logistic Regression Performance ===\n")
cat("Accuracy:", round(log_cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(log_cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(log_cm$byClass["Specificity"], 4), "\n")
cat("AUC:", round(auc(log_roc), 4), "\n")
print(log_cm$table)

# ============================================
# 2. SUPPORT VECTOR MACHINE (SVM)
# ============================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SUPPORT VECTOR MACHINE\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# SVM with hyperparameter tuning
svm_grid <- expand.grid(sigma = c(0.01, 0.05, 0.1, 0.5),
                        C = c(0.5, 1, 2, 5))
install.packages("kernlab")
library(kernlab)
svm_model <- train(target ~ .,
                   data = train_scaled,
                   method = "svmRadial",
                   trControl = cv_control,
                   tuneGrid = svm_grid,
                   metric = "ROC")

print(svm_model)

# Predictions
svm_pred <- predict(svm_model, newdata = test_scaled)
svm_prob <- predict(svm_model, newdata = test_scaled, type = "prob")
# Performance
svm_cm <- confusionMatrix(svm_pred, y_test)
svm_roc <- roc(y_test, svm_prob$Diseased)

cat("\n=== SVM Performance ===\n")
cat("Accuracy:", round(svm_cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(svm_cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(svm_cm$byClass["Specificity"], 4), "\n")
cat("AUC:", round(auc(svm_roc), 4), "\n")
cat("\nBest parameters:\n")
cat("Sigma:", svm_model$bestTune$sigma, "\n")
cat("C:", svm_model$bestTune$C, "\n")

# ============================================
# 3. DECISION TREE
# ============================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("DECISION TREE\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Use unscaled data for decision tree
train_unscaled <- cbind(X_train, target = y_train)
test_unscaled <- cbind(X_test, target = y_test)

# Decision Tree with complexity parameter tuning
dt_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.005))

dt_model <- train(target ~ .,
                  data = train_unscaled,
                  method = "rpart",
                  trControl = cv_control,
                  tuneGrid = dt_grid,
                  metric = "ROC")

print(dt_model)

# Plot the best decision tree
best_tree <- dt_model$finalModel
rpart.plot(best_tree, 
           main = "Decision Tree for Hepatic Disorder",
           extra = 104,
           box.palette = "RdBu",
           fallen.leaves = TRUE)

# Print tree rules
cat("\nDecision Tree Rules:\n")
print(best_tree)

# Predictions
dt_pred <- predict(dt_model, newdata = test_unscaled)
dt_prob <- predict(dt_model, newdata = test_unscaled, type = "prob")

# Performance
dt_cm <- confusionMatrix(dt_pred, y_test)
dt_roc <- roc(y_test, dt_prob$Diseased)

cat("\n=== Decision Tree Performance ===\n")
cat("Accuracy:", round(dt_cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(dt_cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(dt_cm$byClass["Specificity"], 4), "\n")
cat("AUC:", round(auc(dt_roc), 4), "\n")
cat("Best CP:", dt_model$bestTune$cp, "\n")

# ============================================
# 4. MANUAL 5-FOLD CROSS-VALIDATION
# ============================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("MANUAL 5-FOLD CROSS-VALIDATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Function for manual k-fold CV
run_manual_cv <- function(data, k = 5) {
  
  folds <- createFolds(data$target, k = k, list = TRUE)
  
  results <- data.frame(
    Fold = 1:k,
    Log_Accuracy = NA,
    Log_AUC = NA,
    SVM_Accuracy = NA,
    SVM_AUC = NA,
    DT_Accuracy = NA,
    DT_AUC = NA
  )
  
  for(i in 1:k) {
    # Split data
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:nrow(data), test_idx)
    
    train_fold <- data[train_idx, ]
    test_fold <- data[test_idx, ]
    
    # Prepare features
    X_train_fold <- train_fold[, !names(train_fold) %in% c("target")]
    y_train_fold <- train_fold$target
    X_test_fold <- test_fold[, !names(test_fold) %in% c("target")]
    y_test_fold <- test_fold$target
    
    # Scale
    preprocess_fold <- preProcess(X_train_fold, method = c("center", "scale"))
    X_train_scaled_fold <- predict(preprocess_fold, X_train_fold)
    X_test_scaled_fold <- predict(preprocess_fold, X_test_fold)
    
    train_fold_scaled <- cbind(X_train_scaled_fold, target = y_train_fold)
    test_fold_scaled <- cbind(X_test_scaled_fold, target = y_test_fold)
    
    # Logistic Regression
    log_fold <- glm(target ~ ., data = train_fold_scaled, family = binomial)
    log_prob_fold <- predict(log_fold, newdata = test_fold_scaled, type = "response")
    log_pred_fold <- ifelse(log_prob_fold > 0.5, "Diseased", "Healthy")
    log_pred_fold <- factor(log_pred_fold, levels = c("Healthy", "Diseased"))
    
    log_cm_fold <- confusionMatrix(log_pred_fold, y_test_fold)
    log_roc_fold <- roc(y_test_fold, log_prob_fold, quiet = TRUE)
    
    # SVM
    svm_fold <- svm(target ~ ., data = train_fold_scaled, 
                    kernel = "radial", probability = TRUE)
    svm_prob_fold <- attr(predict(svm_fold, test_fold_scaled, probability = TRUE), 
                          "probabilities")[, "Diseased"]
    svm_pred_fold <- predict(svm_fold, test_fold_scaled)
    
    svm_cm_fold <- confusionMatrix(svm_pred_fold, y_test_fold)
    svm_roc_fold <- roc(y_test_fold, svm_prob_fold, quiet = TRUE)
    
    # Decision Tree
    train_fold_unscaled <- cbind(X_train_fold, target = y_train_fold)
    test_fold_unscaled <- cbind(X_test_fold, target = y_test_fold)
    
    dt_fold <- rpart(target ~ ., data = train_fold_unscaled, method = "class")
    dt_prob_fold <- predict(dt_fold, test_fold_unscaled, type = "prob")[, "Diseased"]
    dt_pred_fold <- predict(dt_fold, test_fold_unscaled, type = "class")
    
    dt_cm_fold <- confusionMatrix(dt_pred_fold, y_test_fold)
    dt_roc_fold <- roc(y_test_fold, dt_prob_fold, quiet = TRUE)
    
    # Store results
    results$Log_Accuracy[i] <- log_cm_fold$overall["Accuracy"]
    results$Log_AUC[i] <- as.numeric(auc(log_roc_fold))
    results$SVM_Accuracy[i] <- svm_cm_fold$overall["Accuracy"]
    results$SVM_AUC[i] <- as.numeric(auc(svm_roc_fold))
    results$DT_Accuracy[i] <- dt_cm_fold$overall["Accuracy"]
    results$DT_AUC[i] <- as.numeric(auc(dt_roc_fold))
  }
  
  return(results)
}

# Run manual CV
cv_results <- run_manual_cv(train_data, k = 5)

print(cv_results)

cat("\n=== Cross-Validation Summary ===\n")
cat("Logistic Regression - Mean Accuracy:", round(mean(cv_results$Log_Accuracy), 4), 
    "±", round(sd(cv_results$Log_Accuracy), 4), "\n")
cat("Logistic Regression - Mean AUC:", round(mean(cv_results$Log_AUC), 4), 
    "±", round(sd(cv_results$Log_AUC), 4), "\n\n")

cat("SVM - Mean Accuracy:", round(mean(cv_results$SVM_Accuracy), 4), 
    "±", round(sd(cv_results$SVM_Accuracy), 4), "\n")
cat("SVM - Mean AUC:", round(mean(cv_results$SVM_AUC), 4), 
    "±", round(sd(cv_results$SVM_AUC), 4), "\n\n")

cat("Decision Tree - Mean Accuracy:", round(mean(cv_results$DT_Accuracy), 4), 
    "±", round(sd(cv_results$DT_Accuracy), 4), "\n")
cat("Decision Tree - Mean AUC:", round(mean(cv_results$DT_AUC), 4), 
    "±", round(sd(cv_results$DT_AUC), 4), "\n")

# ============================================
# MODEL COMPARISON
# ============================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("MODEL COMPARISON TABLE\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

comparison <- data.frame(
  Model = c("Logistic Regression", "SVM", "Decision Tree"),
  Accuracy = c(log_cm$overall["Accuracy"],
               svm_cm$overall["Accuracy"],
               dt_cm$overall["Accuracy"]),
  Sensitivity = c(log_cm$byClass["Sensitivity"],
                  svm_cm$byClass["Sensitivity"],
                  dt_cm$byClass["Sensitivity"]),
  Specificity = c(log_cm$byClass["Specificity"],
                  svm_cm$byClass["Specificity"],
                  dt_cm$byClass["Specificity"]),
  AUC = c(auc(log_roc), auc(svm_roc), auc(dt_roc)),
  CV_Accuracy = c(mean(cv_results$Log_Accuracy),
                  mean(cv_results$SVM_Accuracy),
                  mean(cv_results$DT_Accuracy)),
  CV_AUC = c(mean(cv_results$Log_AUC),
             mean(cv_results$SVM_AUC),
             mean(cv_results$DT_AUC))
)

print(comparison)

# ============================================
# ROC CURVES COMPARISON
# ============================================
plot(log_roc, col = "blue", lwd = 2, 
     main = "ROC Curves for Hepatic Disorder Prediction",
     print.auc = FALSE)
plot(svm_roc, col = "red", lwd = 2, add = TRUE)
plot(dt_roc, col = "green", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(paste("Logistic (AUC:", round(auc(log_roc), 3), ")"),
                  paste("SVM (AUC:", round(auc(svm_roc), 3), ")"),
                  paste("Decision Tree (AUC:", round(auc(dt_roc), 3), ")")),
       col = c("blue", "red", "green"), lwd = 2)

# ============================================
# FEATURE IMPORTANCE
# ============================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("DECISION TREE FEATURE IMPORTANCE\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

dt_importance <- varImp(dt_model)
print(dt_importance)

# Plot feature importance
plot(dt_importance, main = "Decision Tree Feature Importance for Hepatic Disorder")

# ============================================
# SAVE RESULTS
# ============================================
saveRDS(log_model, "logistic_model.rds")
saveRDS(svm_model, "svm_model.rds")
saveRDS(dt_model, "decision_tree_model.rds")
saveRDS(preprocess_params, "preprocess_params.rds")

write.csv(comparison, "model_comparison.csv", row.names = FALSE)
write.csv(cv_results, "cross_validation_results.csv", row.names = FALSE)

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("ANALYSIS COMPLETE!\n")
cat("Models saved: logistic_model.rds, svm_model.rds, decision_tree_model.rds\n")
cat("Results saved: model_comparison.csv, cross_validation_results.csv\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# ============================================
# PREDICTION FUNCTION FOR CLINICAL USE
# ============================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("PREDICTION FUNCTION EXAMPLE\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Create prediction function
predict_hepatic <- function(age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio, 
                            model = log_model, model_type = "logistic") {
  
  # Create data frame for new patient
  new_patient <- data.frame(
    Age = age,
    Gender = gender,
    TB = tb,
    DB = db,
    ALKPHOS = alkphos,
    SGPT = sgpt,
    SGOT = sgot,
    TP = tp,
    ALB = alb,
    AG_RATIO = ag_ratio
  )
  
  # Scale the data
  scaled_patient <- predict(preprocess_params, new_patient)
  
  # Make prediction
  if(model_type %in% c("logistic", "svm")) {
    prob <- predict(model, newdata = scaled_patient, type = "prob")$Diseased
  } else {
    prob <- predict(model, newdata = new_patient, type = "prob")$Diseased
  }
  
  prediction <- ifelse(prob > 0.5, "Diseased", "Healthy")
  
  result <- data.frame(
    Prediction = prediction,
    Risk_Probability = round(prob, 3),
    Clinical_Recommendation = ifelse(prob > 0.7, "HIGH RISK - Immediate hepatology referral recommended",
                                     ifelse(prob > 0.5, "MODERATE RISK - Further liver function tests advised",
                                            "LOW RISK - Routine monitoring sufficient"))
  )
  
  return(result)
}

# Example prediction
cat("\nExample prediction for a 45-year-old male patient:\n")
example_result <- predict_hepatic(
  age = 45, gender = 1, tb = 1.5, db = 0.6, alkphos = 250,
  sgpt = 45, sgot = 50, tp = 6.5, alb = 3.5, ag_ratio = 0.9,
  model = log_model, model_type = "logistic"
)
print(example_result)

cat("\nScript completed successfully!\n")

# After your comparison, identify the best model
best_model_name <- comparison$Model[which.max(comparison$AUC)]
best_model <- switch(best_model_name,
                     "Logistic Regression" = log_model,
                     "SVM" = svm_model,
                     "Decision Tree" = dt_model)

cat("\n=== BEST MODEL SELECTED ===\n")
cat("Model:", best_model_name, "\n")
cat("AUC:", max(comparison$AUC), "\n")
cat("Accuracy:", comparison$Accuracy[which.max(comparison$AUC)], "\n")

# Save the best model for deployment
saveRDS(best_model, "best_hepatic_model.rds")
# ============================================
# CLINICAL INTERPRETATION OF BEST MODEL
# ============================================

# For Logistic Regression - Get Odds Ratios
if(best_model_name == "Logistic Regression") {
  cat("\n=== CLINICAL INTERPRETATION (Logistic Regression) ===\n")
  
  # Get coefficients
  coef_summary <- summary(log_model$finalModel)$coefficients
  odds_ratios <- data.frame(
    Biomarker = rownames(coef_summary)[-1],
    Coefficient = coef_summary[-1, 1],
    Odds_Ratio = exp(coef_summary[-1, 1]),
    CI_lower = exp(coef_summary[-1, 1] - 1.96 * coef_summary[-1, 2]),
    CI_upper = exp(coef_summary[-1, 1] + 1.96 * coef_summary[-1, 2]),
    P_value = coef_summary[-1, 4]
  )
  
  # Sort by odds ratio
  odds_ratios <- odds_ratios[order(abs(odds_ratios$Odds_Ratio - 1), decreasing = TRUE), ]
  print(odds_ratios)
  
  # Clinical summary
  cat("\n=== KEY CLINICAL FINDINGS ===\n")
  for(i in 1:min(5, nrow(odds_ratios))) {
    if(odds_ratios$Odds_Ratio[i] > 1) {
      cat(odds_ratios$Biomarker[i], ": ", round(odds_ratios$Odds_Ratio[i], 2), 
          "x increased risk (p=", format(odds_ratios$P_value[i], scientific = TRUE), ")\n", sep="")
    } else {
      cat(odds_ratios$Biomarker[i], ": ", round(odds_ratios$Odds_Ratio[i], 2), 
          "x decreased risk (p=", format(odds_ratios$P_value[i], scientific = TRUE), ")\n", sep="")
    }
  }
}

# For Decision Tree - Extract Rules
if(best_model_name == "Decision Tree") {
  cat("\n=== CLINICAL DECISION RULES ===\n")
  
  # Extract decision rules
  tree_text <- capture.output(print(best_tree))
  cat("Clinical Decision Tree Rules:\n")
  cat(tree_text, sep = "\n")
  
  # Identify key splitting variables
  frame <- best_tree$frame
  used_vars <- unique(row.names(frame)[frame$var != "<leaf>"])
  cat("\nKey Clinical Biomarkers in Decision Tree:\n")
  print(used_vars)
}

# For SVM - Get Variable Importance
if(best_model_name == "SVM") {
  cat("\n=== SVM VARIABLE IMPORTANCE ===\n")
  svm_importance <- varImp(svm_model)
  print(svm_importance)
  
  # Top 5 important biomarkers
  top5 <- rownames(svm_importance$importance)[order(svm_importance$importance$Overall, 
                                                    decreasing = TRUE)][1:5]
  cat("\nTop 5 Most Important Biomarkers for SVM:\n")
  print(top5)
}

# ============================================
# FIND OPTIMAL CLINICAL THRESHOLD
# ============================================

# Get probabilities from best model
if(best_model_name == "Logistic Regression") {
  best_prob <- predict(best_model, newdata = test_scaled, type = "prob")$Diseased
} else if(best_model_name == "SVM") {
  best_prob <- predict(best_model, newdata = test_scaled, type = "prob")$Diseased
} else {
  best_prob <- predict(best_model, newdata = test_unscaled, type = "prob")$Diseased
}

# Calculate Youden's Index for different thresholds
thresholds <- seq(0.1, 0.9, by = 0.05)
youden <- data.frame(threshold = thresholds, 
                     sensitivity = NA, 
                     specificity = NA,
                     youden_index = NA)

for(i in 1:length(thresholds)) {
  pred_class <- factor(ifelse(best_prob > thresholds[i], "Diseased", "Healthy"),
                       levels = c("Healthy", "Diseased"))
  cm <- confusionMatrix(pred_class, y_test)
  youden$sensitivity[i] <- cm$byClass["Sensitivity"]
  youden$specificity[i] <- cm$byClass["Specificity"]
  youden$youden_index[i] <- youden$sensitivity[i] + youden$specificity[i] - 1
}

# Optimal threshold
optimal_thresh <- youden$threshold[which.max(youden$youden_index)]
optimal_youden <- max(youden$youden_index, na.rm = TRUE)

cat("\n=== OPTIMAL CLINICAL THRESHOLD ===\n")
cat("Optimal probability threshold:", optimal_thresh, "\n")
cat("Youden's Index at optimal threshold:", round(optimal_youden, 3), "\n")
cat("Sensitivity:", youden$sensitivity[which.max(youden$youden_index)], "\n")
cat("Specificity:", youden$specificity[which.max(youden$youden_index)], "\n")

# Plot threshold optimization
plot(thresholds, youden$sensitivity, type = "l", col = "blue", 
     ylim = c(0,1), xlab = "Probability Threshold", 
     ylab = "Metric Value", lwd = 2,
     main = "Threshold Optimization for Clinical Decision")
lines(thresholds, youden$specificity, col = "red", lwd = 2)
lines(thresholds, youden$youden_index, col = "green", lwd = 2)
abline(v = optimal_thresh, lty = 2, col = "black")
legend("right", legend = c("Sensitivity", "Specificity", "Youden's Index"),
       col = c("blue", "red", "green"), lwd = 2)

# ============================================
# SUBGROUP ANALYSIS
# ============================================

# Age groups
test_unscaled$age_group <- cut(test_unscaled$Age, 
                               breaks = c(0, 30, 50, 70, 100),
                               labels = c("Young (<30)", "Middle (30-50)", 
                                          "Senior (50-70)", "Elderly (>70)"))

# Performance by age group
age_performance <- data.frame()

for(age in levels(test_unscaled$age_group)) {
  idx <- which(test_unscaled$age_group == age)
  if(length(idx) > 0) {
    prob_subset <- best_prob[idx]
    true_subset <- y_test[idx]
    
    # Calculate AUC for this age group
    if(length(unique(true_subset)) > 1) {
      auc_subset <- roc(true_subset, prob_subset, quiet = TRUE)$auc
      age_performance <- rbind(age_performance, 
                               data.frame(Age_Group = age, 
                                          N = length(idx),
                                          AUC = as.numeric(auc_subset)))
    }
  }
}

cat("\n=== PERFORMANCE BY AGE GROUP ===\n")
print(age_performance)

# Performance by gender
gender_performance <- data.frame()

for(g in c(0, 1)) {
  idx <- which(test_unscaled$Gender == g)
  prob_subset <- best_prob[idx]
  true_subset <- y_test[idx]
  
  if(length(unique(true_subset)) > 1) {
    auc_subset <- roc(true_subset, prob_subset, quiet = TRUE)$auc
    gender_performance <- rbind(gender_performance,
                                data.frame(Gender = ifelse(g == 0, "Female", "Male"),
                                           N = length(idx),
                                           AUC = as.numeric(auc_subset)))
  }
}

cat("\n=== PERFORMANCE BY GENDER ===\n")
print(gender_performance)

# Statistical test for age difference
if(nrow(age_performance) > 1) {
  cat("\nNote: Age subgroup differences should be interpreted cautiously due to sample size variations.\n")
}


# ============================================
# CALIBRATION PLOT (How reliable are predictions?)
# ============================================

# Create calibration data
calibration_data <- data.frame(
  predicted_prob = best_prob,
  observed = as.numeric(y_test == "Diseased")
)

# Create bins
calibration_data$prob_bin <- cut(calibration_data$predicted_prob, 
                                 breaks = seq(0, 1, by = 0.1))
library(tidyverse)
# Calculate observed frequency in each bin
calibration_summary <- calibration_data %>%
  group_by(prob_bin) %>%
  summarise(
    n = n(),
    mean_pred = mean(predicted_prob),
    mean_obs = mean(observed),
    se = sqrt(mean_obs * (1 - mean_obs) / n)
  )

# Plot calibration
ggplot(calibration_summary, aes(x = mean_pred, y = mean_obs)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_obs - 1.96*se, ymax = mean_obs + 1.96*se), width = 0.02) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Model Calibration Plot",
       x = "Predicted Probability",
       y = "Observed Proportion") +
  theme_minimal() +
  xlim(0,1) + ylim(0,1)

cat("\nCalibration assessment: Points close to diagonal line indicate good calibration\n")


# Print final summary for your report
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("FINAL SUMMARY FOR MSC THESIS\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

cat("Best Performing Model:", best_model_name, "\n")
cat("AUC:", round(max(comparison$AUC), 3), "\n")
cat("Accuracy:", round(comparison$Accuracy[which.max(comparison$AUC)], 3), "\n\n")

cat("Top 3 Most Important Clinical Biomarkers:\n")
if(best_model_name == "Logistic Regression") {
  for(i in 1:3) {
    cat("  ", i, ".", odds_ratios$Biomarker[i], 
        "(OR =", round(odds_ratios$Odds_Ratio[i], 2), ")\n", sep="")
  }
} else if(best_model_name == "Decision Tree") {
  for(i in 1:min(3, length(used_vars))) {
    cat("  ", i, ".", used_vars[i], "\n", sep="")
  }
} else {
  top_imp <- rownames(svm_importance$importance)[order(svm_importance$importance$Overall, 
                                                       decreasing = TRUE)][1:3]
  for(i in 1:3) {
    cat("  ", i, ".", top_imp[i], "\n", sep="")
  }
}

cat("\nClinical Decision Threshold:", round(optimal_thresh, 2), "\n")
cat("\n✅ Analysis complete! Results saved for your thesis.\n")

































