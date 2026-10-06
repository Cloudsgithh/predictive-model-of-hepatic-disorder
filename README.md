# ============================================================================
# MODEL COMPARISON: Logistic Regression, Decision Tree, Random Forest, SVM
# ============================================================================

# Load required libraries
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(pROC)
library(ggplot2)
library(dplyr)

# ============================================================================
# 1. DATA PREPARATION (Assuming 'train_scaled' and 'test_scaled' exist)
# ============================================================================
data <- read.csv("C:\\Users\\Dell\\Downloads\\cleaned_liver_data.csv")

# Ensure target variable is a factor
train_scaled$SELECTOR <- as.factor(train_scaled$SELECTOR)
test_scaled$SELECTOR <- as.factor(test_scaled$SELECTOR)

# ============================================================================
# 2. TRAIN ALL MODELS
# ============================================================================

set.seed(123)

# --- 2.1 Logistic Regression ---
cat("\n=== TRAINING LOGISTIC REGRESSION ===\n")
log_model <- glm(
  SELECTOR ~ .,
  data = train_scaled,
  family = binomial(link = "logit")
)
summary(log_model)

# --- 2.2 Decision Tree ---
cat("\n=== TRAINING DECISION TREE ===\n")
dt_model <- rpart(
  SELECTOR ~ .,
  data = train_scaled,
  method = "class",
  control = rpart.control(
    minsplit = 20,
    minbucket = 7,
    maxdepth = 10,
    cp = 0.01
  )
)

# Prune the tree using cross-validation
printcp(dt_model)
dt_model_pruned <- prune(dt_model, cp = dt_model$cptable[which.min(dt_model$cptable[, "xerror"]), "CP"])
cat("Optimal CP:", dt_model$cptable[which.min(dt_model$cptable[, "xerror"]), "CP"], "\n")

# --- 2.3 Random Forest ---
cat("\n=== TRAINING RANDOM FOREST ===\n")
rf_model <- randomForest(
  SELECTOR ~ .,
  data = train_scaled,
  ntree = 500,
  mtry = sqrt(ncol(train_scaled) - 1),
  importance = TRUE,
  na.action = na.omit
)
print(rf_model)

# --- 2.4 SVM with RBF Kernel ---
cat("\n=== TRAINING SVM ===\n")
svm_model <- svm(
  SELECTOR ~ .,
  data = train_scaled,
  kernel = "radial",
  cost = 1,
  gamma = 0.1,
  probability = TRUE
)
print(svm_model)

# ============================================================================
# 3. MAKE PREDICTIONS
# ============================================================================

# --- 3.1 Logistic Regression Predictions ---
log_pred <- predict(log_model, test_scaled, type = "response")
log_class <- ifelse(log_pred > 0.5, "Disorder", "Normal")
log_class <- factor(log_class, levels = c("Normal", "Disorder"))

# --- 3.2 Decision Tree Predictions ---
dt_pred <- predict(dt_model_pruned, test_scaled, type = "class")
dt_prob <- predict(dt_model_pruned, test_scaled, type = "prob")[, "Disorder"]

# --- 3.3 Random Forest Predictions ---
rf_pred <- predict(rf_model, test_scaled)
rf_prob <- predict(rf_model, test_scaled, type = "prob")[, "Disorder"]

# --- 3.4 SVM Predictions ---
svm_pred <- predict(svm_model, test_scaled)
svm_prob <- attr(predict(svm_model, test_scaled, probability = TRUE), "probabilities")[, "Disorder"]

# ============================================================================
# 4. EVALUATION FUNCTION
# ============================================================================

calculate_metrics <- function(actual, predicted, probabilities = NULL) {
  # Confusion matrix
  cm <- confusionMatrix(predicted, actual)
  
  # Metrics
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- cm$byClass["F1"]
  
  # AUC
  if (!is.null(probabilities)) {
    roc_obj <- roc(actual, probabilities, quiet = TRUE)
    auc_value <- auc(roc_obj)
  } else {
    auc_value <- NA
  }
  
  return(c(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1 = f1,
    AUC = auc_value
  ))
}

# ============================================================================
# 5. CALCULATE METRICS FOR ALL MODELS
# ============================================================================

metrics_log <- calculate_metrics(test_scaled$SELECTOR, log_class, log_pred)
metrics_dt <- calculate_metrics(test_scaled$SELECTOR, dt_pred, dt_prob)
metrics_rf <- calculate_metrics(test_scaled$SELECTOR, rf_pred, rf_prob)
metrics_svm <- calculate_metrics(test_scaled$SELECTOR, svm_pred, svm_prob)

# ============================================================================
# 6. CREATE COMPARISON TABLE
# ============================================================================

comparison_df <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "SVM"),
  Accuracy = c(metrics_log["Accuracy"], metrics_dt["Accuracy"], 
               metrics_rf["Accuracy"], metrics_svm["Accuracy"]),
  Precision = c(metrics_log["Precision"], metrics_dt["Precision"], 
                metrics_rf["Precision"], metrics_svm["Precision"]),
  Recall = c(metrics_log["Recall"], metrics_dt["Recall"], 
             metrics_rf["Recall"], metrics_svm["Recall"]),
  F1_Score = c(metrics_log["F1"], metrics_dt["F1"], 
               metrics_rf["F1"], metrics_svm["F1"]),
  AUC = c(metrics_log["AUC"], metrics_dt["AUC"], 
          metrics_rf["AUC"], metrics_svm["AUC"])
)

# Format for display
comparison_display <- comparison_df
comparison_display$Accuracy <- round(comparison_display$Accuracy * 100, 2)
comparison_display$Precision <- round(comparison_display$Precision * 100, 2)
comparison_display$Recall <- round(comparison_display$Recall * 100, 2)
comparison_display$F1_Score <- round(comparison_display$F1_Score * 100, 2)
comparison_display$AUC <- round(comparison_display$AUC, 4)

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("MODEL COMPARISON RESULTS\n")
cat("=", rep("=", 70), "\n\n", sep = "")
print(comparison_display)

# ============================================================================
# 7. IDENTIFY BEST MODEL
# ============================================================================

# Find best model based on multiple criteria
best_auc <- comparison_df[which.max(comparison_df$AUC), ]
best_f1 <- comparison_df[which.max(comparison_df$F1_Score), ]
best_acc <- comparison_df[which.max(comparison_df$Accuracy), ]

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("BEST PERFORMING MODELS\n")
cat("=", rep("=", 70), "\n\n", sep = "")

cat("Best by AUC:\n")
print(best_auc[, c("Model", "AUC", "F1_Score", "Accuracy")])

cat("\nBest by F1-Score:\n")
print(best_f1[, c("Model", "F1_Score", "AUC", "Accuracy")])

cat("\nBest by Accuracy:\n")
print(best_acc[, c("Model", "Accuracy", "AUC", "F1_Score")])

# ============================================================================
# 8. ROC CURVES
# ============================================================================

# Generate ROC objects
roc_log <- roc(test_scaled$SELECTOR, log_pred, quiet = TRUE)
roc_dt <- roc(test_scaled$SELECTOR, dt_prob, quiet = TRUE)
roc_rf <- roc(test_scaled$SELECTOR, rf_prob, quiet = TRUE)
roc_svm <- roc(test_scaled$SELECTOR, svm_prob, quiet = TRUE)

# Plot ROC curves
plot(roc_log, col = "green", lwd = 2, 
     main = "ROC Curves Comparison - All Models",
     xlab = "1 - Specificity",
     ylab = "Sensitivity")

plot(roc_rf, col = "blue", lwd = 2, add = TRUE)
plot(roc_svm, col = "red", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(
         paste("Logistic (AUC =", round(auc(roc_log), 3), ")"),
         paste("Decision Tree (AUC =", round(auc(roc_dt), 3), ")"),
         paste("RF (AUC =", round(auc(roc_rf), 3), ")"),
         paste("SVM (AUC =", round(auc(roc_svm), 3), ")")
       ),
       col = c("green", "orange", "blue", "red"),
       lwd = 2,
       cex = 0.8)

# Load libraries
library(ggplot2)
library(pROC)
library(dplyr)

# Create data frame for plotting
roc_data <- data.frame(
  fpr = c(1 - roc_log$specificities, 
          1 - roc_rf$specificities, 
          1 - roc_svm$specificities),
  tpr = c(roc_log$sensitivities, 
          roc_rf$sensitivities, 
          roc_svm$sensitivities),
  model = c(
    rep("Logistic (AUC = 0.705)", length(roc_log$sensitivities)),
    rep("RF (AUC = 0.699)", length(roc_rf$sensitivities)),
    rep("SVM (AUC = 0.676)", length(roc_svm$sensitivities))
  )
)

# Create ggplot
ggplot(roc_data, aes(x = fpr, y = tpr, color = model)) +
  geom_line(size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "gray50", size = 0.5) +
  scale_color_manual(
    values = c("Logistic (AUC = 0.705)" = "green",
               "RF (AUC = 0.699)" = "blue",
               "SVM (AUC = 0.676)" = "red")
  ) +
  labs(
    title = "ROC Curves Comparison - All Models",
    x = "1 - Specificity",
    y = "Sensitivity",
    color = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    legend.position = c(0.7, 0.2),
    legend.background = element_rect(fill = "white", color = "black", size = 0.2),
    panel.grid.minor = element_blank()
  ) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  annotate("text", x = 0.7, y = 0.1, 
           label = paste("Logistic AUC = 0.705"), 
           color = "green", size = 3.5) +
  annotate("text", x = 0.7, y = 0.05, 
           label = paste("RF AUC = 0.699"), 
           color = "blue", size = 3.5) +
  annotate("text", x = 0.7, y = 0.0, 
           label = paste("SVM AUC = 0.676"), 
           color = "red", size = 3.5)

# ============================================================================
# 9. CONFUSION MATRICES
# ============================================================================

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("CONFUSION MATRICES\n")
cat("=", rep("=", 70), "\n\n", sep = "")

cat("Logistic Regression:\n")
print(confusionMatrix(log_class, test_scaled$SELECTOR)$table)

cat("\nDecision Tree:\n")
print(confusionMatrix(dt_pred, test_scaled$SELECTOR)$table)

cat("\nRandom Forest:\n")
print(confusionMatrix(rf_pred, test_scaled$SELECTOR)$table)

cat("\nSVM:\n")
print(confusionMatrix(svm_pred, test_scaled$SELECTOR)$table)

# ============================================================================
# 10. VISUALIZATION - Bar Plot Comparison
# ============================================================================


library(ggplot2)

# Create data directly (replace values with your actual metrics)
data <- data.frame(
  Model = rep(c("Logistic Regression", "Random Forest", "SVM"), each = 4),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1_Score"), 4),
  Value = c(
    68.24, 71.43, 82.64, 76.62,  # Logistic Regression
    66.47, 74.62, 80.17, 77.29,  # Random Forest
    71.76, 74.17, 92.56, 82.35   # SVM
  )
)

# Create plot
ggplot(data, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  labs(
    title = "Model Performance Comparison",
    x = "Model",
    y = "Score (%)",
    fill = "Metric"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
    legend.position = "bottom"
  ) +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, by = 10),
    labels = function(x) paste0(x, "%")
  ) +
  geom_text(
    aes(label = paste0(round(Value, 1), "%")),
    position = position_dodge(width = 0.8),
    vjust = -0.3,
    size = 2.0
  )

library(ggplot2)

# Create data directly (3 models × 4 metrics = 12 values)
data <- data.frame(
  Model = rep(c("Logistic Regression", "Random Forest", "SVM"), each = 4),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1_Score"), times = 3),
  Value = c(
    68.24, 71.43, 82.64, 76.62,  # Logistic Regression
    66.47, 74.62, 80.17, 77.29,  # Random Forest
    71.76, 74.17, 92.56, 82.35   # SVM
  )
)

# Check structure
print(data)

# Create bar chart
ggplot(data, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  labs(
    title = "Model Performance Comparison",
    subtitle = "All Models (Decision Tree Excluded)",
    x = "Model",
    y = "Score (%)",
    fill = "Metric"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "gray50"),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
    legend.position = "bottom",
    legend.title = element_text(face = "bold")
  ) +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, by = 10),
    labels = function(x) paste0(x, "%")
  ) +
  geom_text(
    aes(label = paste0(round(Value, 1), "%")),
    position = position_dodge(width = 0.8),
    vjust = -0.3,
    size = 3.5,
    fontface = "bold"
  )



# ============================================================================
# 11. RANKING SUMMARY
# ============================================================================

# Rank models by each metric
rank_summary <- comparison_df %>%
  mutate(
    AUC_Rank = rank(-AUC),
    F1_Rank = rank(-F1_Score),
    Accuracy_Rank = rank(-Accuracy),
    Recall_Rank = rank(-Recall),
    Precision_Rank = rank(-Precision)
  ) %>%
  mutate(
    Avg_Rank = round((AUC_Rank + F1_Rank + Accuracy_Rank + Recall_Rank + Precision_Rank) / 5, 2)
  ) %>%
  arrange(Avg_Rank)

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("MODEL RANKING SUMMARY\n")
cat("=", rep("=", 70), "\n\n", sep = "")
print(rank_summary[, c("Model", "AUC", "F1_Score", "Accuracy", 
                       "AUC_Rank", "F1_Rank", "Accuracy_Rank", "Avg_Rank")])

# ============================================================================
# 12. FINAL RECOMMENDATION
# ============================================================================

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("FINAL RECOMMENDATION\n")
cat("=", rep("=", 70), "\n\n", sep = "")

best_model <- rank_summary[which.min(rank_summary$Avg_Rank), "Model"]
best_auc_val <- round(max(comparison_df$AUC), 4)
best_f1_val <- round(max(comparison_df$F1_Score) * 100, 2)
best_acc_val <- round(max(comparison_df$Accuracy) * 100, 2)

cat("Based on comprehensive evaluation across all metrics,\n")
cat("the BEST performing model is:", best_model, "\n\n")

cat("Performance Summary of Best Model:\n")
cat("  • AUC:", best_auc_val, "\n")
cat("  • F1-Score:", best_f1_val, "%\n")
cat("  • Accuracy:", best_acc_val, "%\n\n")

cat("RECOMMENDATION:\n")
if (best_model == "Random Forest") {
  cat("✅ Random Forest is recommended due to:\n")
  cat("   - Best discriminative ability (highest AUC)\n")
  cat("   - Balanced performance across all metrics\n")
  cat("   - Robustness to overfitting\n")
  cat("   - Provides variable importance for clinical insights\n")
} else if (best_model == "SVM") {
  cat("✅ SVM is recommended for screening purposes due to:\n")
  cat("   - High sensitivity (detects most Disorder cases)\n")
  cat("   - Strong performance with non-linear relationships\n")
  cat("   - Effective with moderate-sized datasets\n")
} else if (best_model == "Logistic Regression") {
  cat("✅ Logistic Regression is recommended due to:\n")
  cat("   - Highest interpretability\n")
  cat("   - Provides odds ratios for each predictor\n")
  cat("   - Efficient with small to medium datasets\n")
  cat("   - No hyperparameter tuning required\n")
} else if (best_model == "Decision Tree") {
  cat("✅ Decision Tree is recommended for:\n")
  cat("   - Clear visualization of decision rules\n")
  cat("   - Easy explanation to non-technical stakeholders\n")
  cat("   - Handling missing values naturally\n")
}

# ============================================================================
# 13. ADDITIONAL DIAGNOSTICS
# ============================================================================

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("ADDITIONAL INSIGHTS\n")
cat("=", rep("=", 70), "\n\n", sep = "")

# Check if AUC differences are meaningful
cat("AUC Comparison (higher is better):\n")
comparison_df %>%
  arrange(desc(AUC)) %>%
  select(Model, AUC) %>%
  mutate(AUC_Rounded = round(AUC, 4)) %>%
  print()

# Identify trade-offs
cat("\nKey Trade-offs to Consider:\n")
cat("• If priority is SENSITIVITY (detect disorders): Choose model with highest Recall\n")
cat("• If priority is SPECIFICITY (avoid false alarms): Choose model with highest Precision\n")
cat("• If priority is BALANCED PERFORMANCE: Choose model with highest F1-Score\n")
cat("• If priority is PROBABILITY RANKING: Choose model with highest AUC\n")

# ============================================================================
# CHI-SQUARE TEST: GENDER VS LIVER DISEASE
# ============================================================================

# Load data
data <- read.csv("C:\\Users\\Dell\\Downloads\\cleaned_liver_data.csv")

# Rename columns (based on dataset structure)
colnames(data)[1:11] <- c("Age", "Gender", "TB", "DB", "ALKPHOS", "SGPT", 
                          "SGOT", "TP", "ALB", "AG_RATIO", "Selector")

# Remove extra columns and missing values
data <- data[, 1:11]
data <- na.omit(data)

# Recode Selector: 1 = Disease, 2 = Normal
data$Disease <- ifelse(data$Selector == 1, "Disease", "Normal")

# ============================================================================
# CHI-SQUARE TEST
# ============================================================================

# Create contingency table
contingency_table <- table(data$Gender, data$Disease)
print("Contingency Table (Gender vs Disease):")
print(contingency_table)

# Perform Chi-Square test
chi_test <- chisq.test(contingency_table)

# Display results
cat("\n========================================\n")
cat("CHI-SQUARE TEST RESULTS\n")
cat("========================================\n")
cat("Chi-Square Statistic:", chi_test$statistic, "\n")
cat("Degrees of Freedom:", chi_test$parameter, "\n")
cat("P-value:", chi_test$p.value, "\n")

# Interpretation
cat("\n========================================\n")
cat("INTERPRETATION\n")
cat("========================================\n")
if (chi_test$p.value < 0.05) {
  cat("✅ Significant association between Gender and Liver Disease (p < 0.05)\n")
  cat("Gender DOES influence liver disease risk.\n")
} else {
  cat("❌ NO significant association between Gender and Liver Disease (p >= 0.05)\n")
  cat("Gender does NOT influence liver disease risk.\n")
}

# Show proportions
cat("\nProportion with Disease by Gender:\n")
prop.table(contingency_table, 1) * 100
