# ============================================================
# CORRECTED CODE - PREDICTIVE ANALYSIS OF HEPATIC DISORDER
# ============================================================

# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(ggplot2)
library(corrplot)
library(MLmetrics)

# Set seed for reproducibility
set.seed(123)

# ============================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================

# Read the data
data <- read.csv("C:\\Users\\Dell\\Downloads\\cleaned_liver_data.csv")
# CONVERT SELECTOR TO FACTOR WITH VALID NAMES
# This is the key fix - use meaningful names instead of numbers
data$SELECTOR <- factor(data$SELECTOR, 
                        levels = c(1, 2), 
                        labels = c("disorder", "normal"))

# View structure
str(data)
summary(data)

# Check for missing values
sum(is.na(data))



# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

# Target variable distribution
table(data$SELECTOR)
prop.table(table(data$SELECTOR))

# Visualize target distribution
ggplot(data, aes(x = SELECTOR)) +
  geom_bar(fill = c("steelblue", "coral")) +
  labs(title = "Distribution of Hepatic Disorder Status",
       x = "Selector (Normal vs Disorder)",
       y = "Count") +
  theme_minimal()

# Boxplots for key biomarkers
key_vars <- c("TB", "DB", "ALKPHOS", "SGPT", "SGOT", "TP", "ALB")

data %>%
  pivot_longer(cols = all_of(key_vars), 
               names_to = "Biomarker", 
               values_to = "Value") %>%
  ggplot(aes(x = SELECTOR, y = Value, fill = SELECTOR)) +
  geom_boxplot() +
  facet_wrap(~Biomarker, scales = "free_y") +
  labs(title = "Biomarker Distribution by Hepatic Disorder Status",
       x = "Selector (Normal vs Disorder)",
       y = "Value") +
  theme_minimal() +
  theme(legend.position = "none")

# ============================================================
# 3. DATA SPLITTING
# ============================================================

# Split data into training and testing sets (70/30)
set.seed(123)
train_index <- createDataPartition(data$SELECTOR, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")
cat("Training set distribution:\n")
print(table(train_data$SELECTOR))
cat("Test set distribution:\n")
print(table(test_data$SELECTOR))

# ============================================================
# 4. PREPROCESSING - SCALING AND CENTERING
# ============================================================

# Identify numeric predictors (excluding target)
predictors <- c("Age", "TB", "DB", "ALKPHOS", "SGPT", "SGOT", 
                "TP", "ALB", "A.G.RATIO")


# Preprocessing object for scaling
preproc <- preProcess(train_data[, predictors], method = c("center", "scale"))

# Apply preprocessing
train_scaled <- predict(preproc, train_data[, predictors])
test_scaled <- predict(preproc, test_data[, predictors])

# Add back the target variable
train_scaled$SELECTOR <- train_data$SELECTOR
test_scaled$SELECTOR <- test_data$SELECTOR

# ============================================================
# 5. MODEL TRAINING AND EVALUATION FUNCTIONS
# ============================================================

# Function to calculate evaluation metrics
calculate_metrics <- function(actual, predicted, probabilities = NULL) {
  # Confusion matrix
  cm <- confusionMatrix(predicted, actual)
  
  # Accuracy
  accuracy <- cm$overall["Accuracy"]
  
  # Precision, Recall, F1 (for the positive class - "Disorder")
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- cm$byClass["F1"]
  
  # If probabilities are provided, calculate AUC
  if (!is.null(probabilities)) {
    auc_value <- auc(roc(actual, probabilities))
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

# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================

cat("\n", "=", rep("=", 50), "\n", sep = "")
cat("RANDOM FOREST MODEL\n")
cat("=", rep("=", 50), "\n\n", sep = "")

# Train Random Forest
set.seed(123)
rf_model <- randomForest(
  SELECTOR ~ ., 
  data = train_scaled,
  ntree = 500,
  mtry = sqrt(ncol(train_scaled) - 1),
  importance = TRUE,
  na.action = na.omit
)

print(rf_model)

# Variable importance
importance_df <- as.data.frame(importance(rf_model))
importance_df$Variable <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$MeanDecreaseGini), ]


ggplot(importance_df, aes(x = reorder(Variable, MeanDecreaseGini), 
                          y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Random Forest - Variable Importance",
       x = "Variable",
       y = "Mean Decrease Gini") +
  theme_minimal()

# Predictions
rf_pred <- predict(rf_model, test_scaled)
rf_prob <- predict(rf_model, test_scaled, type = "prob")[, "Disorder"]

# Metrics
rf_metrics <- calculate_metrics(test_scaled$SELECTOR, rf_pred, rf_prob)
cat("\nRandom Forest Metrics:\n")
print(rf_metrics)

# ============================================================
# 7. SVM MODEL
# ============================================================

cat("\n", "=", rep("=", 50), "\n", sep = "")
cat("SUPPORT VECTOR MACHINE (SVM) MODEL\n")
cat("=", rep("=", 50), "\n\n", sep = "")

# SVM with radial kernel (tune parameters)
set.seed(123)
svm_tune <- tune(
  svm,
  SELECTOR ~ .,
  data = train_scaled,
  kernel = "radial",
  ranges = list(
    cost = c(0.1, 1, 10, 100),
    gamma = c(0.01, 0.1, 1)
  ),
  tunecontrol = tune.control(cross = 5)
)

cat("Best SVM parameters:\n")
print(svm_tune$best.parameters)

# Train SVM with best parameters
svm_model <- svm(
  SELECTOR ~ .,
  data = train_scaled,
  kernel = "radial",
  cost = svm_tune$best.parameters$cost,
  gamma = svm_tune$best.parameters$gamma,
  probability = TRUE
)

# Predictions
svm_pred <- predict(svm_model, test_scaled)
svm_prob <- attr(predict(svm_model, test_scaled, probability = TRUE), "probabilities")[, "Disorder"]

# Metrics
svm_metrics <- calculate_metrics(test_scaled$SELECTOR, svm_pred, svm_prob)
cat("\nSVM Metrics:\n")
print(svm_metrics)

# ============================================================
# 8. K-FOLD CROSS VALIDATION (for both models)
# ============================================================

cat("\n", "=", rep("=", 50), "\n", sep = "")
cat("K-FOLD CROSS VALIDATION (10-fold)\n")
cat("=", rep("=", 50), "\n\n", sep = "")

# Define cross-validation settings
ctrl <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary  # This handles binary classification properly
)

# Logistic Regression with 10-fold CV
library(caret)

ctrl <- trainControl(
  method = "cv", 
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

set.seed(123)
logistic_cv <- train(
  SELECTOR ~ ., 
  data = train_scaled,
  method = "glm",
  family = binomial,
  trControl = ctrl,
  metric = "ROC"
)

# View results
print(logistic_cv)
# Output includes Accuracy and ROC with SD

# Load required libraries
library(caret)
library(rpart)
library(rpart.plot)

# Set seed for reproducibility
set.seed(123)

# Define cross-validation control
train_control <- trainControl(
  method = "cv",           # Cross-validation
  number = 10,             # 10-fold CV
  classProbs = TRUE,       # Get class probabilities for AUC
  summaryFunction = twoClassSummary  # Get ROC, Sens, Spec
)

# Train Decision Tree with CV
dt_cv <- train(
  SELECTOR ~ .,
  data = train_scaled,
  method = "rpart",
  trControl = train_control,
  tuneLength = 10,         # Test 10 different cp values
  metric = "ROC"
)

# Print results
print(dt_cv)
print(dt_cv$results)




# 8.1 Random Forest CV
cat("\n--- Random Forest 10-fold CV ---\n")
set.seed(123)
rf_cv <- train(
  SELECTOR ~ .,
  data = train_scaled,
  method = "rf",
  trControl = ctrl,
  ntree = 500,
  tuneGrid = data.frame(mtry = c(2, 3, 4, 5)),
  metric = "ROC"
)

print(rf_cv)
cat("\nRandom Forest CV Results:\n")
cat("Best mtry:", rf_cv$bestTune$mtry, "\n")
cat("CV Accuracy:", max(rf_cv$results$Accuracy), "\n")
cat("CV ROC:", max(rf_cv$results$ROC), "\n")

# 8.2 SVM CV
cat("\n--- SVM 10-fold CV ---\n")
set.seed(123)
svm_cv <- train(
  SELECTOR ~ .,
  data = train_scaled,
  method = "svmRadial",
  trControl = ctrl,
  tuneGrid = expand.grid(
    sigma = c(0.01, 0.1, 1),
    C = c(0.1, 1, 10, 100)
  ),
  metric = "ROC"
)

print(svm_cv)
cat("\nSVM CV Results:\n")
cat("Best sigma:", svm_cv$bestTune$sigma, "\n")
cat("Best C:", svm_cv$bestTune$C, "\n")
cat("CV Accuracy:", max(svm_cv$results$Accuracy), "\n")
cat("CV ROC:", max(svm_cv$results$ROC), "\n")

# ============================================================
# 9. COMPARISON OF MODELS
# ============================================================

cat("\n", "=", rep("=", 50), "\n", sep = "")
cat("MODEL COMPARISON RESULTS\n")
cat("=", rep("=", 50), "\n\n", sep = "")

# Create comparison table
comparison_df <- data.frame(
  Model = c("logistic","Random Forest", "SVM"),
  Accuracy = c(logistic_metrics["Accuracy"],rf_metrics["Accuracy"], svm_metrics["Accuracy"]),
  Precision = c(logistic_metrics["Precision"],rf_metrics["Precision"], svm_metrics["Precision"]),
  Recall = c(logistic_metrics["Recall"],rf_metrics["Recall"], svm_metrics["Recall"]),
  F1_Score = c(logistic_metrics["F1"],rf_metrics["F1"], svm_metrics["F1"]),
  AUC = c(logistic_metrics["AUC"],rf_metrics["AUC"], svm_metrics["AUC"])
)

print(comparison_df)



# ============================================================
# 10. ROC CURVES COMPARISON
# ============================================================

# ROC Curves
roc_log <- roc(test_scaled$SELECTOR, log_pred, quiet = TRUE)
rf_roc <- roc(test_scaled$SELECTOR, rf_prob)
svm_roc <- roc(test_scaled$SELECTOR, svm_prob)
plot(roc_log, col = "green", lwd = 2, 
     main = "ROC Curves Comparison - All Models",
     xlab = "1 - Specificity",
     ylab = "Sensitivity")

plot(rf_roc, col = "blue", lwd = 2, 
     main = "ROC Curves Comparison")
plot(svm_roc, col = "red", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(paste("Random Forest (AUC =", round(auc(rf_roc), 3), ")"),
                  paste("SVM (AUC =", round(auc(svm_roc), 3), ")")),
       col = c("blue", "red"), lwd = 2)

# ============================================================
# 11. CONFUSION MATRICES
# ============================================================

# Random Forest Confusion Matrix
cat("\nRandom Forest Confusion Matrix:\n")
print(confusionMatrix(rf_pred, test_scaled$SELECTOR))

# SVM Confusion Matrix
cat("\nSVM Confusion Matrix:\n")

print(confusionMatrix(svm_pred, test_scaled$SELECTOR))

# ============================================================
# 12. VISUALIZATION OF RESULTS
# ============================================================

# Bar plot of model comparison
comparison_long <- comparison_df %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1_Score, AUC),
               names_to = "Metric", values_to = "Value")

ggplot(comparison_long, aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Metric, scales = "free_y") +
  labs(title = "Model Performance Comparison",
       y = "Score",
       x = "Model") +
  theme_minimal() +
  theme(legend.position = "bottom")



















# ============================================
# PREDICTION FUNCTION FOR CLINICAL USE
# ============================================
# ============================================================================
# RECREATE COMPARISON DATA FRAME
# ============================================================================

# Assuming you have these metric objects from earlier
comparison_df <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "SVM"),
  Accuracy = c(
    0.6824,  # Logistic Regression
    0.6529,  # Decision Tree
    0.6647,  # Random Forest
    0.7176   # SVM
  ),
  Precision = c(
    0.7143,  # Logistic Regression
    0.6923,  # Decision Tree
    0.7462,  # Random Forest
    0.7417   # SVM
  ),
  Recall = c(
    0.8264,  # Logistic Regression
    0.8017,  # Decision Tree
    0.8017,  # Random Forest
    0.9256   # SVM
  ),
  F1_Score = c(
    0.7662,  # Logistic Regression
    0.7433,  # Decision Tree
    0.7729,  # Random Forest
    0.8235   # SVM
  ),
  AUC = c(
    0.7047,  # Logistic Regression
    0.5000,  # Decision Tree
    0.6993,  # Random Forest
    0.6758   # SVM
  )
)

# View the data
print(comparison_df)

# Now find the best model
best_model_name <- comparison_df$Model[which.max(comparison_df$AUC)]
cat("\nBest model based on AUC:", best_model_name, "\n")


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
best_model_name <- comparison_df$Model[which.max(comparison_df$AUC)]
best_model <- switch(best_model_name,
                     "Logistic Regression" = log_model,
                     "SVM" = svm_model,
                     "Decision Tree" = dt_model)

cat("\n=== BEST MODEL SELECTED ===\n")
cat("Model:", best_model_name, "\n")
cat("AUC:", max(comparison_df$AUC), "\n")
cat("Accuracy:", comparison_df$Accuracy[which.max(comparison_df$AUC)], "\n")

# Save the best model for deployment
saveRDS(best_model, "best_hepatic_model.rds")


# ============================================================================
# COMPLETE CORRECTED CODE - FIXES ALL ERRORS
# ============================================================================

# Load required libraries
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(ggplot2)

# ============================================================================
# 1. FUNCTION TO GET PREDICTORS FROM MODEL
# ============================================================================

get_predictors <- function(model) {
  predictors <- NULL
  
  if (inherits(model, "train")) {
    if (!is.null(model$trainingData)) {
      predictors <- colnames(model$trainingData)
      predictors <- predictors[predictors != ".outcome"]
    }
  } else if (inherits(model, c("glm", "lm"))) {
    if (!is.null(model$coefficients)) {
      predictors <- names(coef(model))
      predictors <- predictors[predictors != "(Intercept)"]
    }
  } else if (inherits(model, "randomForest")) {
    if (!is.null(model$forest)) {
      predictors <- rownames(model$importance)
    }
  } else if (inherits(model, "svm")) {
    if (!is.null(model$terms)) {
      predictors <- attr(model$terms, "term.labels")
    }
  } else if (inherits(model, "rpart")) {
    if (!is.null(model$terms)) {
      predictors <- attr(model$terms, "term.labels")
    }
  }
  
  return(predictors)
}

# ============================================================================
# 2. FUNCTION TO GET VARIABLE TYPES FROM TRAINING DATA
# ============================================================================

get_variable_types <- function(model) {
  var_types <- list()
  
  if (inherits(model, "train")) {
    if (!is.null(model$trainingData)) {
      training_data <- model$trainingData
      # Remove outcome column
      training_data <- training_data[, !names(training_data) %in% ".outcome", drop = FALSE]
      
      for (col in names(training_data)) {
        var_types[[col]] <- class(training_data[[col]])[1]
      }
    }
  }
  
  return(var_types)
}

# ============================================================================
# 3. CREATE PATIENT DATA WITH CORRECT TYPES
# ============================================================================

create_patient <- function(predictors, var_types = NULL, age = 45) {
  
  # Create empty data frame
  patient <- data.frame(matrix(ncol = length(predictors), nrow = 1))
  colnames(patient) <- predictors
  
  # Default values
  for (var in predictors) {
    # Check if we have type information
    var_type <- if (!is.null(var_types) && var %in% names(var_types)) var_types[[var]] else NULL
    
    # Set values based on variable name and type
    if (grepl("Age", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(age)
      
    } else if (grepl("Gender|Sex", var, ignore.case = TRUE)) {
      # IMPORTANT: Check the expected type
      if (!is.null(var_type) && var_type %in% c("numeric", "integer")) {
        patient[[var]] <- as.numeric(1)  # 1 = Male
      } else {
        patient[[var]] <- factor("Male", levels = c("Male", "Female"))
      }
      
    } else if (grepl("TB|Bilirubin", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(2.5)
      
    } else if (grepl("DB|Direct.*Bilirubin", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(1.0)
      
    } else if (grepl("ALKPHOS|Alkaline", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(180)
      
    } else if (grepl("SGPT|ALT", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(85)
      
    } else if (grepl("SGOT|AST", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(70)
      
    } else if (grepl("TP|Total.*Protein", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(8.5)
      
    } else if (grepl("ALB|Albumin", var, ignore.case = TRUE) && !grepl("ALKPHOS", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(4.0)
      
    } else if (grepl("AG_RATIO|A.G.RATIO|A/G|Albumin.*Globulin", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(0.9)
      
    } else if (grepl("BMI", var, ignore.case = TRUE)) {
      patient[[var]] <- as.numeric(25)
      
    } else if (grepl("Smoking", var, ignore.case = TRUE)) {
      if (!is.null(var_type) && var_type %in% c("numeric", "integer")) {
        patient[[var]] <- as.numeric(0)  # 0 = No
      } else {
        patient[[var]] <- factor("No", levels = c("Yes", "No"))
      }
      
    } else if (grepl("Diabetes", var, ignore.case = TRUE)) {
      if (!is.null(var_type) && var_type %in% c("numeric", "integer")) {
        patient[[var]] <- as.numeric(0)
      } else {
        patient[[var]] <- factor("No", levels = c("Yes", "No"))
      }
      
    } else {
      # Default: set as numeric NA
      patient[[var]] <- as.numeric(NA)
    }
  }
  
  return(patient)
}

# ============================================================================
# 4. PREDICT FUNCTION WITH PROPER ERROR HANDLING
# ============================================================================

predict_patient <- function(model, patient_data, preproc = NULL) {
  
  # Apply scaling if preproc is provided
  patient_scaled <- patient_data
  
  if (!is.null(preproc)) {
    tryCatch({
      # Try to predict with scaling
      patient_scaled <- predict(preproc, patient_data)
    }, error = function(e) {
      message("⚠️ Scaling failed. Using raw data.")
      patient_scaled <- patient_data
    })
  }
  
  # Make prediction based on model type
  if (inherits(model, "train")) {
    # caret train object
    prob <- predict(model, patient_scaled, type = "prob")
    class <- predict(model, patient_scaled, type = "raw")
    
    if (is.data.frame(prob) && "Disorder" %in% colnames(prob)) {
      prob_value <- prob[, "Disorder"]
    } else if (is.data.frame(prob)) {
      prob_value <- prob[, 1]
    } else {
      prob_value <- as.numeric(prob)
    }
    
    return(list(probability = prob_value, class = class))
    
  } else if (inherits(model, c("glm", "lm"))) {
    prob <- predict(model, patient_scaled, type = "response")
    class <- ifelse(prob > 0.5, "Disorder", "Normal")
    return(list(probability = prob, class = class))
    
  } else if (inherits(model, "randomForest")) {
    prob <- predict(model, patient_scaled, type = "prob")
    class <- predict(model, patient_scaled)
    
    if (is.matrix(prob) && "Disorder" %in% colnames(prob)) {
      prob_value <- prob[, "Disorder"]
    } else if (is.matrix(prob)) {
      prob_value <- prob[, 1]
    } else {
      prob_value <- as.numeric(prob)
    }
    
    return(list(probability = prob_value, class = class))
    
  } else if (inherits(model, "svm")) {
    prob_attr <- attr(predict(model, patient_scaled, probability = TRUE), "probabilities")
    class <- predict(model, patient_scaled)
    
    if (is.matrix(prob_attr) && "Disorder" %in% colnames(prob_attr)) {
      prob_value <- prob_attr[, "Disorder"]
    } else if (is.matrix(prob_attr)) {
      prob_value <- prob_attr[, 1]
    } else {
      prob_value <- as.numeric(prob_attr)
    }
    
    return(list(probability = prob_value, class = class))
    
  } else if (inherits(model, "rpart")) {
    prob <- predict(model, patient_scaled, type = "prob")
    class <- predict(model, patient_scaled, type = "class")
    
    if (is.matrix(prob) && "Disorder" %in% colnames(prob)) {
      prob_value <- prob[, "Disorder"]
    } else if (is.matrix(prob)) {
      prob_value <- prob[, 1]
    } else {
      prob_value <- as.numeric(prob)
    }
    
    return(list(probability = prob_value, class = class))
    
  } else {
    stop("Unsupported model type: ", class(model)[1])
  }
}

# ============================================================================
# 5. PREDICT ALL MODELS
# ============================================================================

predict_all_models <- function(models_list, patient, preproc = NULL) {
  
  cat("\n", "=", rep("=", 70), "\n", sep = "")
  cat("ALL MODELS COMPARISON FOR 45-YEAR-OLD MALE\n")
  cat("=", rep("=", 70), "\n\n", sep = "")
  
  results <- data.frame(
    Model = character(),
    Probability = numeric(),
    Class = character(),
    Recommendation = character(),
    stringsAsFactors = FALSE
  )
  
  for (model_name in names(models_list)) {
    model <- models_list[[model_name]]
    
    if (is.null(model)) {
      next
    }
    
    cat("Predicting with:", model_name, "...\n")
    
    tryCatch({
      result <- predict_patient(model, patient, preproc)
      
      new_row <- data.frame(
        Model = model_name,
        Probability = round(result$probability * 100, 1),
        Class = as.character(result$class),
        Recommendation = ifelse(result$class == "Disorder", 
                                "Further evaluation", "Routine follow-up"),
        stringsAsFactors = FALSE
      )
      results <- rbind(results, new_row)
      cat("  ✓ Success\n")
      
    }, error = function(e) {
      cat("  ✗ Error:", e$message, "\n")
      new_row <- data.frame(
        Model = model_name,
        Probability = NA,
        Class = "Error",
        Recommendation = paste("Error:", e$message),
        stringsAsFactors = FALSE
      )
      results <- rbind(results, new_row)
    })
  }
  
  return(results)
}

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("PATIENT PREDICTION FOR 45-YEAR-OLD MALE\n")
cat("=", rep("=", 70), "\n\n", sep = "")

# Step 1: Get predictors
cat("Step 1: Extracting predictors...\n")
predictors <- get_predictors(log_model)

if (is.null(predictors) || length(predictors) == 0) {
  cat("⚠️ Could not extract predictors. Using manual list.\n")
  predictors <- c("Age", "TB", "DB", "ALKPHOS", "SGPT", "SGOT", "TP", "ALB", "AG_RATIO")
}
cat("✓ Predictors:", paste(predictors, collapse = ", "), "\n")

# Step 2: Get variable types from training data
cat("\nStep 2: Getting variable types...\n")
var_types <- get_variable_types(log_model)
if (length(var_types) > 0) {
  cat("✓ Found variable types for", length(var_types), "variables\n")
}

# Step 3: Create patient data with correct types
cat("\nStep 3: Creating patient data...\n")
patient <- create_patient(predictors, var_types, age = 45)
cat("✓ Patient data created\n")

# Show patient data structure
cat("\nPatient data structure:\n")
str(patient)

# Step 4: Create model list
cat("\nStep 4: Creating model list...\n")
all_models <- list()

if (exists("log_model") && !is.null(log_model)) {
  all_models[["Logistic Regression"]] <- log_model
  cat("✓ Added Logistic Regression\n")
}

if (exists("rf_model") && !is.null(rf_model)) {
  all_models[["Random Forest"]] <- rf_model
  cat("✓ Added Random Forest\n")
}

if (exists("svm_model") && !is.null(svm_model)) {
  all_models[["SVM"]] <- svm_model
  cat("✓ Added SVM\n")
}

if (exists("dt_model") && !is.null(dt_model)) {
  all_models[["Decision Tree"]] <- dt_model
  cat("✓ Added Decision Tree\n")
} else if (exists("dt_model_pruned") && !is.null(dt_model_pruned)) {
  all_models[["Decision Tree"]] <- dt_model_pruned
  cat("✓ Added Decision Tree (pruned)\n")
}

# Step 5: Make predictions
cat("\nStep 5: Making predictions...\n")
results <- predict_all_models(all_models, patient, preproc)

# Step 6: Display results
cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("PREDICTION RESULTS\n")
cat("=", rep("=", 70), "\n\n", sep = "")
print(results)

# Step 7: Visualization
valid_results <- results[!is.na(results$Probability) & results$Class != "Error", ]
# ============================================================================
# COMPLETE DIAGNOSTIC + PLOT CODE
# ============================================================================

library(ggplot2)

# 1. DIAGNOSTIC: Check your data
cat("=", rep("=", 50), "\n", sep = "")
cat("DATA DIAGNOSTIC\n")
cat("=", rep("=", 50), "\n\n", sep = "")

cat("Original results data:\n")
print(results)

cat("\nFiltered valid results:\n")
print(valid_results)

cat("\nStructure of valid_results:\n")
str(valid_results)

cat("\nUnique Class values:", paste(unique(valid_results$Class), collapse = ", "), "\n")

# 2. CREATE PLOT WITH SAFE COLOR HANDLING
cat("\n", "=", rep("=", 50), "\n", sep = "")
cat("CREATING PLOT\n")
cat("=", rep("=", 50), "\n\n", sep = "")

# Create the base plot
p <- ggplot(valid_results, aes(x = Model, y = Probability, fill = as.factor(Class))) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  labs(
    title = "Disorder Probability for 45-Year-Old Male Patient",
    x = "Model",
    y = "Probability of Disorder (%)",
    fill = "Predicted Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title = element_text(face = "bold", size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
    axis.text.y = element_text(size = 10),
    legend.position = "bottom",
    legend.title = element_text(face = "bold")
  ) +
  scale_y_continuous(
    limits = c(0, 100), 
    breaks = seq(0, 100, by = 10),
    labels = function(x) paste0(x, "%")
  ) +
  geom_text(
    aes(label = paste0(round(Probability, 1), "%")),
    position = position_dodge(width = 0.8),
    vjust = -0.3,
    size = 4.5,
    fontface = "bold"
  ) +
  # Use a safe color palette that works with any class values
  scale_fill_brewer(palette = "Set2")

# Display
print(p)


# Step 8: Clinical interpretation
cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("CLINICAL INTERPRETATION\n")
cat("=", rep("=", 70), "\n\n", sep = "")

if ("Logistic Regression" %in% results$Model) {
  lr_result <- results[results$Model == "Logistic Regression", ]
  if (!is.na(lr_result$Probability) && lr_result$Class != "Error") {
    cat("Patient: 45-year-old male with elevated liver enzymes\n")
    cat("Logistic Regression Probability:", lr_result$Probability, "%\n")
    
    if (lr_result$Probability > 70) {
      cat("→ HIGH RISK: Clinical evaluation strongly recommended\n")
    } else if (lr_result$Probability > 40) {
      cat("→ MODERATE RISK: Consider additional tests\n")
    } else {
      cat("→ LOW RISK: Routine follow-up recommended\n")
    }
  }
}

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("END\n")
cat("=", rep("=", 70), "\n", sep = "")































# ============================================================================
# COMPLETE CORRECTED CODE FOR PATIENT PREDICTION
# ============================================================================

# Load required libraries
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(ggplot2)

# ============================================================================
# 1. FIX: GET PREDICTORS FROM MODEL
# ============================================================================

get_predictors <- function(model) {
  predictors <- NULL
  
  # For caret train objects
  if (inherits(model, "train")) {
    if (!is.null(model$trainingData)) {
      predictors <- colnames(model$trainingData)
      predictors <- predictors[predictors != ".outcome"]
    }
  }
  
  # For glm objects
  if (is.null(predictors) && inherits(model, c("glm", "lm"))) {
    if (!is.null(model$coefficients)) {
      predictors <- names(coef(model))
      predictors <- predictors[predictors != "(Intercept)"]
    }
  }
  
  # For randomForest
  if (is.null(predictors) && inherits(model, "randomForest")) {
    if (!is.null(model$forest)) {
      predictors <- rownames(model$importance)
    }
  }
  
  # For svm
  if (is.null(predictors) && inherits(model, "svm")) {
    if (!is.null(model$terms)) {
      predictors <- attr(model$terms, "term.labels")
    }
  }
  
  # For rpart
  if (is.null(predictors) && inherits(model, "rpart")) {
    if (!is.null(model$terms)) {
      predictors <- attr(model$terms, "term.labels")
    }
  }
  
  return(predictors)
}

# ============================================================================
# 2. FIX: CREATE PATIENT DATA WITH PROPER TYPES
# ============================================================================

create_patient <- function(predictors, age = 45, gender = "Male") {
  
  # Default values - ensure proper data types
  default_values <- list(
    Age = as.numeric(age),
    Gender = as.factor(gender),
    TB = as.numeric(2.5),
    DB = as.numeric(1.0),
    ALKPHOS = as.numeric(180),
    SGPT = as.numeric(85),
    SGOT = as.numeric(70),
    TP = as.numeric(8.5),
    ALB = as.numeric(4.0),
    AG_RATIO = as.numeric(0.9),  # Note: different name
    A.G.RATIO = as.numeric(0.9),  # Alternative name
    BMI = as.numeric(25),
    Smoking = as.factor("No"),
    Diabetes = as.factor("No")
  )
  
  # Create empty data frame
  patient <- data.frame(matrix(ncol = length(predictors), nrow = 1))
  colnames(patient) <- predictors
  
  # Fill with default values
  for (var in predictors) {
    if (var %in% names(default_values)) {
      patient[[var]] <- default_values[[var]]
    } else {
      # Try to match with similar names
      var_clean <- gsub("[^A-Za-z]", "", var)  # Remove special characters
      for (default_var in names(default_values)) {
        if (var_clean == gsub("[^A-Za-z]", "", default_var)) {
          patient[[var]] <- default_values[[default_var]]
          break
        }
      }
      # If still not found, set to NA with proper type
      if (is.null(patient[[var]]) || is.na(patient[[var]])) {
        patient[[var]] <- NA
      }
    }
  }
  
  return(patient)
}

# ============================================================================
# 3. FIX: PREDICT FUNCTION WITH PROPER ERROR HANDLING
# ============================================================================

predict_patient <- function(model, patient_data, preproc = NULL) {
  
  # Apply scaling if preproc is provided
  patient_scaled <- patient_data
  if (!is.null(preproc)) {
    tryCatch({
      patient_scaled <- predict(preproc, patient_data)
    }, error = function(e) {
      message("⚠️ Scaling failed. Using raw data.")
    })
  }
  
  # Make prediction based on model type
  if (inherits(model, "train")) {
    # caret train object
    prob <- predict(model, patient_scaled, type = "prob")
    class <- predict(model, patient_scaled, type = "raw")
    # Find the correct column for Disorder
    if ("Disorder" %in% colnames(prob)) {
      prob_value <- prob[, "Disorder"]
    } else {
      prob_value <- prob[, 1]  # Use first column
    }
    return(list(probability = prob_value, class = class))
    
  } else if (inherits(model, c("glm", "lm"))) {
    # glm or lm object
    prob <- predict(model, patient_scaled, type = "response")
    class <- ifelse(prob > 0.5, "Disorder", "Normal")
    return(list(probability = prob, class = class))
    
  } else if (inherits(model, "randomForest")) {
    # randomForest object
    prob <- predict(model, patient_scaled, type = "prob")
    class <- predict(model, patient_scaled)
    if ("Disorder" %in% colnames(prob)) {
      prob_value <- prob[, "Disorder"]
    } else {
      prob_value <- prob[, 1]
    }
    return(list(probability = prob_value, class = class))
    
  } else if (inherits(model, "svm")) {
    # svm object
    prob_attr <- attr(predict(model, patient_scaled, probability = TRUE), "probabilities")
    class <- predict(model, patient_scaled)
    if ("Disorder" %in% colnames(prob_attr)) {
      prob_value <- prob_attr[, "Disorder"]
    } else {
      prob_value <- prob_attr[, 1]
    }
    return(list(probability = prob_value, class = class))
    
  } else if (inherits(model, "rpart")) {
    # rpart object
    prob <- predict(model, patient_scaled, type = "prob")
    class <- predict(model, patient_scaled, type = "class")
    if (ncol(prob) >= 2 && "Disorder" %in% colnames(prob)) {
      prob_value <- prob[, "Disorder"]
    } else {
      prob_value <- prob[, 1]
    }
    return(list(probability = prob_value, class = class))
    
  } else {
    stop("Unsupported model type: ", class(model)[1])
  }
}

# ============================================================================
# 4. FIX: PREDICT ALL MODELS FUNCTION
# ============================================================================

predict_all_models <- function(models_list, patient, preproc = NULL) {
  
  cat("\n", "=", rep("=", 70), "\n", sep = "")
  cat("ALL MODELS COMPARISON FOR 45-YEAR-OLD MALE\n")
  cat("=", rep("=", 70), "\n\n", sep = "")
  
  results <- data.frame(
    Model = character(),
    Probability = numeric(),
    Class = character(),
    Recommendation = character(),
    stringsAsFactors = FALSE
  )
  
  for (model_name in names(models_list)) {
    model <- models_list[[model_name]]
    
    if (is.null(model)) {
      next  # Skip NULL models
    }
    
    tryCatch({
      result <- predict_patient(model, patient, preproc)
      
      new_row <- data.frame(
        Model = model_name,
        Probability = round(result$probability * 100, 1),
        Class = as.character(result$class),
        Recommendation = ifelse(result$class == "Disorder", 
                                "Further evaluation", "Routine follow-up"),
        stringsAsFactors = FALSE
      )
      results <- rbind(results, new_row)
      
    }, error = function(e) {
      message(paste("Error predicting with", model_name, ":", e$message))
      new_row <- data.frame(
        Model = model_name,
        Probability = NA,
        Class = "Error",
        Recommendation = paste("Error:", e$message),
        stringsAsFactors = FALSE
      )
      results <- rbind(results, new_row)
    })
  }
  
  return(results)
}

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("PATIENT PREDICTION FOR 45-YEAR-OLD MALE\n")
cat("=", rep("=", 70), "\n\n", sep = "")

# Step 1: Get predictors from logistic regression model
cat("Step 1: Extracting predictors...\n")
predictors <- get_predictors(log_model)

if (is.null(predictors) || length(predictors) == 0) {
  cat("⚠️ Could not extract predictors. Using manual list.\n")
  predictors <- c("Age", "TB", "DB", "ALKPHOS", "SGPT", "SGOT", "TP", "ALB", "AG_RATIO")
}
cat("✓ Predictors:", paste(predictors, collapse = ", "), "\n")

# Step 2: Create patient data
cat("\nStep 2: Creating patient data...\n")
patient <- create_patient(predictors, age = 45, gender = "Male")
cat("✓ Patient data created\n")
View(patient)
# Step 3: Create model list (only include models that exist)
all_models <- list()

if (exists("log_model") && !is.null(log_model)) {
  all_models[["Logistic Regression"]] <- log_model
  cat("✓ Added Logistic Regression\n")
}

if (exists("rf_model") && !is.null(rf_model)) {
  all_models[["Random Forest"]] <- rf_model
  cat("✓ Added Random Forest\n")
}

if (exists("svm_model") && !is.null(svm_model)) {
  all_models[["SVM"]] <- svm_model
  cat("✓ Added SVM\n")
}

# Check for decision tree (different possible names)
if (exists("dt_model") && !is.null(dt_model)) {
  all_models[["Decision Tree"]] <- dt_model
  cat("✓ Added Decision Tree\n")
} else if (exists("dt_model_pruned") && !is.null(dt_model_pruned)) {
  all_models[["Decision Tree"]] <- dt_model_pruned
  cat("✓ Added Decision Tree (pruned)\n")
} else if (exists("decision_tree_model") && !is.null(decision_tree_model)) {
  all_models[["Decision Tree"]] <- decision_tree_model
  cat("✓ Added Decision Tree\n")
}

# Step 4: Make predictions with all models
cat("\nStep 4: Making predictions...\n")
results <- predict_all_models(all_models, patient, preproc)

# Step 5: Display results
cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("PREDICTION RESULTS SUMMARY\n")
cat("=", rep("=", 70), "\n\n", sep = "")

print(results)

# ============================================================================
# 6. VISUALIZATION (if any predictions succeeded)
# ============================================================================

# Filter out failed predictions
valid_results <- results[!is.na(results$Probability) & results$Class != "Error", ]

if (nrow(valid_results) > 0) {
  
  library(ggplot2)
  
  # Create bar plot
  p <- ggplot(valid_results, aes(x = Model, y = Probability, fill = Class)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
    labs(
      title = "Disorder Probability Predictions for 45-Year-Old Male Patient",
      x = "Model",
      y = "Probability of Disorder (%)",
      fill = "Predicted Class"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    ) +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 10)) +
    geom_text(
      aes(label = paste0(round(Probability, 1), "%")),
      position = position_dodge(width = 0.8),
      vjust = -0.3,
      size = 4
    ) +
    scale_fill_manual(values = c("Normal" = "#66C2A5", "Disorder" = "#FC8D62"))
  
  print(p)
  
  # Save plot
  ggsave("patient_predictions.png", width = 10, height = 6, dpi = 300)
  cat("\n✅ Plot saved as: patient_predictions.png\n")
}

# ============================================================================
# 7. CLINICAL INTERPRETATION
# ============================================================================

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("CLINICAL INTERPRETATION\n")
cat("=", rep("=", 70), "\n\n", sep = "")

cat("Patient: 45-year-old male with elevated liver enzymes\n")
cat("-" , rep("-", 50), "\n", sep = "")

# Get Logistic Regression probability
if ("Logistic Regression" %in% results$Model) {
  lr_result <- results[results$Model == "Logistic Regression", ]
  if (!is.na(lr_result$Probability) && lr_result$Class != "Error") {
    cat("• Logistic Regression Probability:", lr_result$Probability, "%\n")
    
    if (lr_result$Probability > 70) {
      cat("  → HIGH RISK: Further clinical evaluation strongly recommended\n")
      cat("  → Possible liver disorder requiring immediate attention\n")
    } else if (lr_result$Probability > 40) {
      cat("  → MODERATE RISK: Consider additional tests\n")
      cat("  → Monitor liver function regularly\n")
    } else {
      cat("  → LOW RISK: Routine follow-up recommended\n")
      cat("  → Continue regular health checkups\n")
    }
  }
}

cat("\n📋 RECOMMENDED CLINICAL ACTION:\n")
cat("  1. Perform comprehensive liver function tests\n")
cat("  2. Evaluate for possible hepatitis or cirrhosis\n")
cat("  3. Consider ultrasound or imaging studies\n")
cat("  4. Review patient's medication history\n")
cat("  5. Schedule follow-up in 3-6 months\n")

cat("\n", "=", rep("=", 70), "\n", sep = "")
cat("END OF PREDICTION\n")
cat("=", rep("=", 70), "\n", sep = "")

