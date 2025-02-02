library(rpart)
library(rpart.plot)
library(ROCR)
library(ggplot2)
library(randomForest)
library(gbm)
library(MASS)
library(Rtsne) 
library(flexclust)


### Week 1 ###
# Step 1: Read in the Data
HMEQ_Loss <- read.csv("HMEQ_Loss.csv")
str(HMEQ_Loss)
summary(HMEQ_Loss)
head(HMEQ_Loss, 6)

# Step 2: Box-Whisker Plots
par(mfrow=c(2, 3))
boxplot(LOAN ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "LOAN")
boxplot(MORTDUE ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "MORTDUE")
boxplot(VALUE ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "VALUE")
boxplot(YOJ ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "YOJ")
boxplot(DEROG ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "DEROG")
boxplot(DELINQ ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "DELINQ")
par(mfrow=c(2, 3))
boxplot(CLAGE ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "CLAGE")
boxplot(NINQ ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "NINQ")
boxplot(CLNO ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "CLNO")
boxplot(DEBTINC ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "DEBTINC")
# Reset the plotting layout to default
par(mfrow=c(1, 1))

# Step 3: Histograms
# Histogram with density line
hist(HMEQ_Loss$LOAN, breaks=50, main="Histogram of LOAN", xlab="LOAN Amount", freq=FALSE)
dens <- density(na.omit(HMEQ_Loss$LOAN))
lines(dens, col="red") # Superimpose density line

# Step 4: Impute "Fix" all the numeric variables that have missing values
# Set missing Target variables to zero
HMEQ_Loss$TARGET_LOSS_AMT[is.na(HMEQ_Loss$TARGET_LOSS_AMT)] <- 0
# Impute numeric variables with missing values using median and create flags
# LOAN has no missing values as it's the loan amount requested
# MORTDUE
HMEQ_Loss$IMP_MORTDUE <- ifelse(is.na(HMEQ_Loss$MORTDUE), median(HMEQ_Loss$MORTDUE, na.rm = TRUE), HMEQ_Loss$MORTDUE)
HMEQ_Loss$M_MORTDUE <- ifelse(is.na(HMEQ_Loss$MORTDUE), 1, 0)
HMEQ_Loss$MORTDUE <- NULL # Delete the original MORTDUE variable
# complex imputation for VALUE 
# First, compute the median VALUE for each JOB category
median_values_by_job <- aggregate(VALUE ~ JOB, data = HMEQ_Loss, FUN = median, na.rm = TRUE)
# Then, impute missing VALUE based on the median of the corresponding JOB category
# Create a new IMP_VALUE variable with imputed values
HMEQ_Loss$IMP_VALUE <- HMEQ_Loss$VALUE
# Loop over the JOB categories to impute missing VALUEs
for (job in median_values_by_job$JOB) {
  # Get the median value for the current JOB
  median_value <- median_values_by_job[median_values_by_job$JOB == job, ]$VALUE
  # Apply the imputation for records with the current JOB and a missing VALUE
  missing_indices <- is.na(HMEQ_Loss$VALUE) & HMEQ_Loss$JOB == job
  HMEQ_Loss$IMP_VALUE[missing_indices] <- median_value
}
# Create the M_VALUE flag variable
HMEQ_Loss$M_VALUE <- as.integer(is.na(HMEQ_Loss$VALUE))
# Now, impute remaining missing IMP_VALUE with the overall median of VALUE, if any
overall_median <- median(HMEQ_Loss$VALUE, na.rm = TRUE)
still_missing <- is.na(HMEQ_Loss$IMP_VALUE)
HMEQ_Loss$IMP_VALUE[still_missing] <- overall_median
HMEQ_Loss$VALUE <- NULL # Delete the original VALUE variable
# YOJ
HMEQ_Loss$IMP_YOJ <- ifelse(is.na(HMEQ_Loss$YOJ), median(HMEQ_Loss$YOJ, na.rm = TRUE), HMEQ_Loss$YOJ)
HMEQ_Loss$M_YOJ <- as.integer(is.na(HMEQ_Loss$YOJ))
HMEQ_Loss$YOJ <- NULL  # Delete the original YOJ variable
# DEROG
HMEQ_Loss$IMP_DEROG <- ifelse(is.na(HMEQ_Loss$DEROG), median(HMEQ_Loss$DEROG, na.rm = TRUE), HMEQ_Loss$DEROG)
HMEQ_Loss$M_DEROG <- as.integer(is.na(HMEQ_Loss$DEROG))
HMEQ_Loss$DEROG <- NULL  # Delete the original DEROG variable
# DELINQ
HMEQ_Loss$IMP_DELINQ <- ifelse(is.na(HMEQ_Loss$DELINQ), median(HMEQ_Loss$DELINQ, na.rm = TRUE), HMEQ_Loss$DELINQ)
HMEQ_Loss$M_DELINQ <- as.integer(is.na(HMEQ_Loss$DELINQ))
HMEQ_Loss$DELINQ <- NULL  # Delete the original DELINQ variable
# CLAGE
HMEQ_Loss$IMP_CLAGE <- ifelse(is.na(HMEQ_Loss$CLAGE), median(HMEQ_Loss$CLAGE, na.rm = TRUE), HMEQ_Loss$CLAGE)
HMEQ_Loss$M_CLAGE <- as.integer(is.na(HMEQ_Loss$CLAGE))
HMEQ_Loss$CLAGE <- NULL  # Delete the original CLAGE variable
# NINQ
HMEQ_Loss$IMP_NINQ <- ifelse(is.na(HMEQ_Loss$NINQ), median(HMEQ_Loss$NINQ, na.rm = TRUE), HMEQ_Loss$NINQ)
HMEQ_Loss$M_NINQ <- as.integer(is.na(HMEQ_Loss$NINQ))
HMEQ_Loss$NINQ <- NULL  # Delete the original NINQ variabl
# CLNO
HMEQ_Loss$IMP_CLNO <- ifelse(is.na(HMEQ_Loss$CLNO), median(HMEQ_Loss$CLNO, na.rm = TRUE), HMEQ_Loss$CLNO)
HMEQ_Loss$M_CLNO <- as.integer(is.na(HMEQ_Loss$CLNO))
HMEQ_Loss$CLNO <- NULL  # Delete the original CLNO variable
# DEBTINC
HMEQ_Loss$IMP_DEBTINC <- ifelse(is.na(HMEQ_Loss$DEBTINC), median(HMEQ_Loss$DEBTINC, na.rm = TRUE), HMEQ_Loss$DEBTINC)
HMEQ_Loss$M_DEBTINC <- as.integer(is.na(HMEQ_Loss$DEBTINC))
HMEQ_Loss$DEBTINC <- NULL  # Delete the original DEBTINC variable
# After all imputations, run a summary to check for missing values
summary(HMEQ_Loss)
# Compute a sum for all the M_ variables
sum_M_variables <- colSums(HMEQ_Loss[, grepl("^M_", names(HMEQ_Loss))])
print(sum_M_variables)

# Step 5: One Hot Encoding
categoryVars <- sapply(HMEQ_Loss, is.character)
for (var in names(HMEQ_Loss)[categoryVars]) {
  levels <- unique(HMEQ_Loss[[var]])
  for (level in levels) {
    HMEQ_Loss[[paste0("FLAG_", var, "_", level)]] <- ifelse(HMEQ_Loss[[var]] == level, 1, 0)
  }
  HMEQ_Loss[[var]] <- NULL # Remove original variable
}
summary(HMEQ_Loss) # Show the dataset after encoding


### Week 2 ###
# Step 1: Read in the Data
# Read the provided scrubbed data into R
HMEQ_Scrubbed <- read.csv("HMEQ_Scrubbed.csv")
df <- HMEQ_Scrubbed

# List the structure of the data
str(df)

# Execute a summary of the data
summary(df)

# Print the first six records
head(df)

# Step 2: Classification Decision Tree
# Set the control parameters for the tree depth and method of splitting
control_parameters <- rpart.control(maxdepth = 10)

# Exclude TARGET_LOSS_AMT from the predictors
df_classification <- df
df_classification$TARGET_LOSS_AMT <- NULL

# Decision tree using Gini
treeGini <- rpart(TARGET_BAD_FLAG ~ ., data = df_classification, method = "class", parms = list(split = "gini"), control = control_parameters)
rpart.plot(treeGini)
print(treeGini$variable.importance)

# Decision tree using Entropy
treeEntropy <- rpart(TARGET_BAD_FLAG ~ ., data = df_classification, method = "class", parms = list(split = "information"), control = control_parameters)
rpart.plot(treeEntropy)
print(treeEntropy$variable.importance)

# Creating ROC curve and calculating AUC for the Gini model
predictionsGini <- predict(treeGini, df_classification, type = "prob")[,2]
rocGini <- prediction(predictionsGini, df_classification$TARGET_BAD_FLAG)
perfGini <- performance(rocGini, "tpr", "fpr")
aucGini <- performance(rocGini, measure = "auc")@y.values[[1]]
plot(perfGini, col = "red")
abline(0, 1, lty = 2)
text(0.6, 0.4, paste("AUC Gini =", round(aucGini, 4)))

# Creating ROC curve and calculating AUC for the Entropy model
predictionsEntropy <- predict(treeEntropy, df_classification, type = "prob")[,2]
rocEntropy <- prediction(predictionsEntropy, df_classification$TARGET_BAD_FLAG)
perfEntropy <- performance(rocEntropy, "tpr", "fpr")
aucEntropy <- performance(rocEntropy, measure = "auc")@y.values[[1]]
plot(perfEntropy, col = "blue")
abline(0, 1, lty = 2)
text(0.6, 0.4, paste("AUC Entropy =", round(aucEntropy, 4)))

# Summary and recommendation
#The decision trees for predicting loan default were built using Gini and Entropy criteria. The Gini tree, with a slightly higher AUC, is the preferred model as it better differentiates between defaults. Key predictors for default include missing debt-to-income information and past delinquency, suggesting that individuals with incomplete financial profiles or a history of late payments are at higher risk. The Gini tree's better performance and interpretability make it the recommended choice.


# Step 3: Regression Decision Tree
# Exclude TARGET_BAD_FLAG from the predictors for the regression tasks
df_regression <- df
df_regression$TARGET_BAD_FLAG <- NULL

# Decision tree using ANOVA
treeAnova <- rpart(TARGET_LOSS_AMT ~ ., data = df_regression, method = "anova", control = control_parameters)
rpart.plot(treeAnova)
print(treeAnova$variable.importance)

# Decision tree using Poisson
treePoisson <- rpart(TARGET_LOSS_AMT ~ ., data = df_regression, method = "poisson", control = control_parameters)
rpart.plot(treePoisson)
print(treePoisson$variable.importance)

# Calculate RMSE for the ANOVA model
predictionsAnova <- predict(treeAnova, df_regression)
RMSE_Anova <- sqrt(mean((df_regression$TARGET_LOSS_AMT - predictionsAnova)^2))
print(RMSE_Anova)

# Calculate RMSE for the Poisson model
predictionsPoisson <- predict(treePoisson, df_regression)
RMSE_Poisson <- sqrt(mean((df_regression$TARGET_LOSS_AMT - predictionsPoisson)^2))
print(RMSE_Poisson)

# Summary and recommendation
#The regression decision trees, using ANOVA and Poisson methods, aimed to predict the loss amount in the event of a loan default. The ANOVA tree, which achieved a lower RMSE, is favored for its more accurate loss predictions. The loan amount was identified as a key determinant of loss, with larger loans likely incurring greater losses. This is logically consistent with lending practices. Thus, the ANOVA tree is recommended, as it sensibly captures the relationship between loan attributes and potential losses.

# Step 4: Probability / Severity Model Decision Tree
# Re-plotting the classification tree using Gini for TARGET_BAD_FLAG (probability of default)
rpart.plot(treeGini)
print(treeGini$variable.importance)

# Building and plotting the decision tree for TARGET_LOSS_AMT for defaulted records only
df_loss <- df[df$TARGET_BAD_FLAG == 1, ]
treeLoss <- rpart(TARGET_LOSS_AMT ~ ., data = df_loss, method = "anova", control = control_parameters)
rpart.plot(treeLoss)
print(treeLoss$variable.importance)

# Predicting the loss given default
loss_given_default <- predict(treeLoss, df_loss)

# Predicting the probability of default using the classification tree
prob_default <- predict(treeGini, df, type = "prob")[,2] 

# Initialize expected loss with zeros
expected_loss <- rep(0, nrow(df))

# Only calculate expected loss where TARGET_BAD_FLAG is 1
expected_loss[df$TARGET_BAD_FLAG == 1] <- prob_default[df$TARGET_BAD_FLAG == 1] * loss_given_default

# Calculate the RMSE value for the Probability / Severity model
# The RMSE will only be calculated for records with TARGET_BAD_FLAG == 1
RMSE_ProbSeverity <- sqrt(mean((df_loss$TARGET_LOSS_AMT - expected_loss[df$TARGET_BAD_FLAG == 1])^2))
print(RMSE_ProbSeverity)

# Summary and recommendation
# The combined Probability/Severity model, integrating the probability of default with the predicted loss amount, yielded a higher RMSE compared to the standalone regression model from Step 3. Despite this, it provides a more holistic risk assessment by considering both the likelihood and impact of loan default. While the standalone ANOVA model from Step 3 is more accurate for predicting losses among defaults, the comprehensive nature of the Probability/Severity model may offer greater practical value for risk management purposes. Therefore, for strategic decision-making, the Probability/Severity model is recommended, despite its lower precision, because it encapsulates the full spectrum of credit risk.

### Week 3 ###
# Step 1: Read in the Data
df <- HMEQ_Scrubbed  

# Step 2: Classification Decision Tree
set.seed(3) # Ensures reproducibility

# Set the control parameters for the tree depth and method of splitting
control_parameters <- rpart.control(maxdepth = 10)

# Exclude TARGET_LOSS_AMT from the predictors
df_classification <- df
df_classification$TARGET_LOSS_AMT <- NULL

# Splitting criteria
FLAG_classification <- sample(c(TRUE, FALSE), nrow(df_classification), replace = TRUE, prob = c(0.7, 0.3))
df_train_classification <- df_classification[FLAG_classification, ]
df_test_classification <- df_classification[!FLAG_classification, ]

# Decision tree using Gini
treeGini <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_classification, method = "class", parms = list(split = "gini"), control = control_parameters)
rpart.plot(treeGini)
print(treeGini$variable.importance)

# Decision tree using Entropy
treeEntropy <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_classification, method = "class", parms = list(split = "information"), control = control_parameters)
rpart.plot(treeEntropy)
print(treeEntropy$variable.importance)

# Gini model on training data
predictionsGini_train <- predict(treeGini, df_train_classification, type = "prob")[,2]
rocGini_train <- prediction(predictionsGini_train, df_train_classification$TARGET_BAD_FLAG)
perfGini_train <- performance(rocGini_train, "tpr", "fpr")
aucGini_train <- performance(rocGini_train, measure = "auc")@y.values[[1]]
plot(perfGini_train, col = "red")
text(0.6, 0.4, paste("AUC Gini Train =", round(aucGini_train, 4)))
# Entropy model on training data
predictionsEntropy_train <- predict(treeEntropy, df_train_classification, type = "prob")[,2]
rocEntropy_train <- prediction(predictionsEntropy_train, df_train_classification$TARGET_BAD_FLAG)
perfEntropy_train <- performance(rocEntropy_train, "tpr", "fpr")
aucEntropy_train <- performance(rocEntropy_train, measure = "auc")@y.values[[1]]
plot(perfEntropy_train, col = "blue")
text(0.6, 0.4, paste("AUC Entropy Train =", round(aucEntropy_train, 4)))
# Gini model on testing data
predictionsGini_test <- predict(treeGini, df_test_classification, type = "prob")[,2]
rocGini_test <- prediction(predictionsGini_test, df_test_classification$TARGET_BAD_FLAG)
perfGini_test <- performance(rocGini_test, "tpr", "fpr")
aucGini_test <- performance(rocGini_test, measure = "auc")@y.values[[1]]
plot(perfGini_test, col = "red")
text(0.6, 0.4, paste("AUC Gini Test =", round(aucGini_test, 4)))
# Entropy model on testing data
predictionsEntropy_test <- predict(treeEntropy, df_test_classification, type = "prob")[,2]
rocEntropy_test <- prediction(predictionsEntropy_test, df_test_classification$TARGET_BAD_FLAG)
perfEntropy_test <- performance(rocEntropy_test, "tpr", "fpr")
aucEntropy_test <- performance(rocEntropy_test, measure = "auc")@y.values[[1]]
plot(perfEntropy_test, col = "blue")
text(0.6, 0.4, paste("AUC Entropy Test =", round(aucEntropy_test, 4)))
# Summary
# The classification decision trees developed to predict loan default using Gini and Entropy show strong performance, with AUC values for both training and testing datasets exceeding 0.8. This suggests that the models are effective at distinguishing between default and non-default cases and are likely neither overfit nor underfit, as indicated by their consistent AUC scores across training and testing sets. The two models perform similarly well, with no significant difference in AUC values. However, the Gini method has a slight edge with marginally higher AUC scores in both training and testing datasets. This could imply that the Gini model has better predictive capability, though the difference is minimal. 


# Step 3: Regression Decision Tree
set.seed(3) # Ensures reproducibility

# Exclude TARGET_BAD_FLAG from the predictors for the regression tasks
df_regression <- df
df_regression$TARGET_BAD_FLAG <- NULL

# Splitting criteria
FLAG_regression <- sample(c(TRUE, FALSE), nrow(df_regression), replace = TRUE, prob = c(0.7, 0.3))
df_train_regression <- df_regression[FLAG_regression, ]
df_test_regression <- df_regression[!FLAG_regression, ]

# Decision tree using ANOVA
treeAnova <- rpart(TARGET_LOSS_AMT ~ ., data = df_train_regression, method = "anova", control = control_parameters)
rpart.plot(treeAnova)
print(treeAnova$variable.importance)

# Decision tree using Poisson
treePoisson <- rpart(TARGET_LOSS_AMT ~ ., data = df_train_regression, method = "poisson", control = control_parameters)
rpart.plot(treePoisson)
print(treePoisson$variable.importance)

# Calculate RMSE for the ANOVA model using training data set
predictionsAnova_train <- predict(treeAnova, df_train_regression)
RMSE_Anova_train <- sqrt(mean((df_train_regression$TARGET_LOSS_AMT - predictionsAnova_train)^2))
print(RMSE_Anova_train)

# Calculate RMSE for the Poisson model using training data set
predictionsPoisson_train <- predict(treePoisson, df_train_regression)
RMSE_Poisson_train <- sqrt(mean((df_train_regression$TARGET_LOSS_AMT - predictionsPoisson_train)^2))
print(RMSE_Poisson_train)

# Calculate RMSE for the ANOVA model using testing data set
predictionsAnova_test <- predict(treeAnova, df_test_regression)
RMSE_Anova_test <- sqrt(mean((df_test_regression$TARGET_LOSS_AMT - predictionsAnova_test)^2))
print(RMSE_Anova_test)

# Calculate RMSE for the Poisson model using testing data set
predictionsPoisson_test <- predict(treePoisson, df_test_regression)
RMSE_Poisson_test <- sqrt(mean((df_test_regression$TARGET_LOSS_AMT - predictionsPoisson_test)^2))
print(RMSE_Poisson_test)

# Summary
# The regression decision trees constructed to predict TARGET_LOSS_AMT reveal that the ANOVA model slightly outperforms the Poisson model, as indicated by lower RMSE values on both training and testing data sets. The RMSE for the ANOVA model is 4824.357 for training and 5263.02 for testing, compared to 5339.834 for training and 5410.932 for testing for the Poisson model. The modest increase in RMSE from training to testing in the ANOVA model suggests it is neither significantly overfit nor underfit, and it generalizes better to unseen data than the Poisson model. The differences in RMSE are not substantial, but they lean in favor of the ANOVA model, indicating it may be more accurate for predicting potential losses on defaulted loans.

# Step 4: Probability / Severity Model Decision Tree
set.seed(3) # Ensuring reproducibility

# Split the full data into training and testing sets
FLAG <- sample(c(TRUE, FALSE), nrow(df), replace = TRUE, prob = c(0.7, 0.3))
df_train <- df[FLAG, ]
df_test <- df[!FLAG, ]

# Exclude TARGET_LOSS_AMT for the classification model training
df_train_excluded <- df_train
df_train_excluded$TARGET_LOSS_AMT <- NULL

# Filter the datasets to include only the instances with a default
df_train_defaults <- df_train[df_train$TARGET_BAD_FLAG == 1, ]
df_test_defaults <- df_test[df_test$TARGET_BAD_FLAG == 1, ]

# Re-train the classification tree using Gini for TARGET_BAD_FLAG (probability of default)
treeGini2 <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_excluded, method = "class", parms = list(split = "gini"), control = control_parameters)
rpart.plot(treeGini2)
print(treeGini2$variable.importance)

# Fit the regression tree for TARGET_LOSS_AMT on df_train_defaults
treeLoss <- rpart(TARGET_LOSS_AMT ~ ., data = df_train_defaults, method = "anova", control = control_parameters)
rpart.plot(treeLoss)
print(treeLoss$variable.importance)

# Use the trained classification model (treeGini2) to predict the probability of default
prob_default_train <- predict(treeGini2, df_train, type = "prob")[,2]
prob_default_test <- predict(treeGini2, df_test, type = "prob")[,2]

# Predict the loss given default using the regression tree
loss_given_default_train <- predict(treeLoss, df_train_defaults)
loss_given_default_test <- predict(treeLoss, df_test_defaults)

# Combine the probability of default with the loss given default to get the expected loss
expected_loss_train <- prob_default_train[df_train$TARGET_BAD_FLAG == 1] * loss_given_default_train
expected_loss_test <- prob_default_test[df_test$TARGET_BAD_FLAG == 1] * loss_given_default_test

# Calculate RMSE for the Probability / Severity model on the training and testing data
RMSE_ProbSeverity_train <- sqrt(mean((df_train_defaults$TARGET_LOSS_AMT - expected_loss_train)^2))
RMSE_ProbSeverity_test <- sqrt(mean((df_test_defaults$TARGET_LOSS_AMT - expected_loss_test)^2))

# Print out the RMSE values
print(paste("RMSE for Probability / Severity model on training data:", RMSE_ProbSeverity_train))
print(paste("RMSE for Probability / Severity model on testing data:", RMSE_ProbSeverity_test))

# Summary
# The Probability/Severity model from Step 4 offers a holistic view of risk by assessing both the likelihood and impact of loan defaults, unlike the direct loss prediction models in Step 3. Despite its higher RMSE, indicating less precision in predicting losses, it provides valuable insights into both default probability and potential loss severity. This comprehensive approach is preferable for risk management, as it balances the incidence and magnitude of risks, making it the recommended choice for a fuller risk assessment.

### Week 4 ###

# Load the data
df <- HMEQ_Scrubbed

# Graph 1: Job Tenure vs. Debt-to-Income Ratio by Loan Status
ggplot(df, aes(x = IMP_YOJ, y = IMP_DEBTINC, color = as.factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("dodgerblue", "tomato"), labels = c("Good Loan", "Bad Loan")) +
  labs(title = "Job Tenure vs. Debt-to-Income Ratio by Loan Status",
       x = "Years on Job",
       y = "Debt-to-Income Ratio (%)",
       color = "Loan Status") +
  theme_classic() +
  geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed")
# This graph shows a comparison between how many years someone has been employed (Years on Job) and their debt-to-income ratio, differentiated by the loan status (good loans in blue, bad loans in red). The dashed line represents a linear trend across all data points, helping to visualize the overall direction of the relationship. By examining where the blue and red dots concentrate, we can see that people who have had stable jobs for many years have a slightly higher likelihood of getting a good loan due to better debt management.

#Graph 2: Loan Amount vs. Mortgage Due by Loan Outcome
ggplot(df, aes(x = IMP_MORTDUE, y = LOAN, color = as.factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c("green", "red"), labels = c("Good Loan", "Bad Loan")) +
  labs(x = "Mortgage Due", y = "Loan Amount", color = "Loan Outcome") +
  theme_minimal() +
  ggtitle("Loan Amount vs. Mortgage Due by Loan Outcome")
#The graph presents a comparison between the home equity loan amounts and the corresponding mortgage dues, with a clear distinction between loans that were paid off (good loans) and those that defaulted (bad loans). While both good and bad loans are distributed across various levels of mortgage dues, good loans are more concentrated at lower mortgage values. Bad loans appear more frequently as mortgage dues increase, suggesting a trend where higher mortgage obligations may correlate with a higher likelihood of loan default. However, the absence of a strong and consistent pattern across the range of mortgage dues indicates that the risk of default is multifactorial and cannot be determined by mortgage dues alone. 

### Week 5 ###

# Step 1: Read in the Data
df <- HMEQ_Scrubbed

# Step 2: Classification Models

set.seed(3) # Ensures reproducibility

# Exclude TARGET_LOSS_AMT from the predictors
df_classification <- df
df_classification$TARGET_LOSS_AMT <- NULL

# Splitting criteria
FLAG_classification <- sample(c(TRUE, FALSE), nrow(df_classification), replace = TRUE, prob = c(0.7, 0.3))
df_train_classification <- df_classification[FLAG_classification, ]
df_test_classification <- df_classification[!FLAG_classification, ]

# TREE
control_parameters <- rpart.control(maxdepth = 10)
tree_model <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_classification, method = "class", control = control_parameters)
rpart.plot(tree_model)
print(tree_model$variable.importance)

pt = predict( tree_model, df_test_classification, type="prob" )
head( pt )
pt2 = prediction( pt[,2], df_test_classification$TARGET_BAD_FLAG)
pt3 = performance( pt2, "tpr", "fpr" )

# RF
rf_model = randomForest(TARGET_BAD_FLAG ~ ., data = df_train_classification, ntree=500, importance=TRUE )
importance( rf_model )
varImpPlot( rf_model )

pr = predict( rf_model, df_test_classification )
head( pr )
pr2 = prediction( pr, df_test_classification$TARGET_BAD_FLAG)
pr3 = performance( pr2, "tpr", "fpr" )

# GRADIENT BOOSTING

gb_model = gbm(TARGET_BAD_FLAG ~ ., data = df_train_classification, n.trees=500, distribution="bernoulli" )
summary.gbm( gb_model, cBars=10 )

pg = predict( gb_model, df_test_classification, type="response" )
head( pg )
pg2 = prediction( pg, df_test_classification$TARGET_BAD_FLAG)
pg3 = performance( pg2, "tpr", "fpr" )

# Plot

plot( pt3, col="green" )
plot( pr3, col="red", add=TRUE )
plot( pg3, col="blue", add=TRUE )
abline(0,1,lty=2)
legend("bottomright",c("TREE","RANDOM FOREST", "GRADIENT BOOSTING"),col=c("green","red","blue"), bty="y", lty=1 )

aucT = performance( pt2, "auc" )@y.values
aucR = performance( pr2, "auc" )@y.values
aucG = performance( pg2, "auc" )@y.values

print( paste("TREE AUC=", aucT) )
print( paste("RF AUC=", aucR) )
print( paste("GB AUC=", aucG) )

# The Random Forest model performed the best, with an AUC of approximately 0.95 which is the highest.
# Considering the ROC curve, where the Random Forest line is closer to the top-left corner compared to the Decision Tree and Gradient Boosting, and the highest AUC value, the Random Forest model has demonstrated superior performance in your classification task. It balances well between bias and variance, can handle a mix of feature types, and is less likely to overfit compared to a single decision tree. Gradient Boosting also performed well and could be a strong contender, especially with further tuning. However, given the evidence, Random Forest would be my primary recommendation for predicting home equity loan defaults in this scenario.

# Step 3: Regression Decision Tree

set.seed(3) # Ensures reproducibility

# Exclude TARGET_BAD_FLAG from the predictors for the regression tasks
df_regression <- df
df_regression$TARGET_BAD_FLAG <- NULL

# Splitting criteria
FLAG_regression <- sample(c(TRUE, FALSE), nrow(df_regression), replace = TRUE, prob = c(0.7, 0.3))
df_train_regression <- df_regression[FLAG_regression, ]
df_test_regression <- df_regression[!FLAG_regression, ]

# TREE

tr_model = rpart( data=df_train_regression, TARGET_LOSS_AMT ~ ., control=control_parameters, method="poisson" )
rpart.plot( tr_model )
rpart.plot( tr_model, digits=-3, extra=100 )
tr_model$variable.importance

pt = predict( tr_model, df_test_regression )
head( pt )
RMSEt = sqrt( mean( ( df_test_regression$TARGET_LOSS_AMT - pt )^2 ) )

# RF

rf_model = randomForest( data=df_train_regression, TARGET_LOSS_AMT ~ ., ntree=500, importance=TRUE )
importance( rf_model )
varImpPlot( rf_model )

pr = predict( rf_model, df_test_regression )
head( pr )
RMSEr = sqrt( mean( ( df_test_regression$TARGET_LOSS_AMT - pr )^2 ) )

# GRADIENT BOOSTING

gb_model = gbm( data=df_train_regression, TARGET_LOSS_AMT ~ ., n.trees=500, distribution="poisson" )
summary.gbm( gb_model, cBars=10 )

pg = predict( gb_model, df_test_regression, type="response" )
head( pg )
RMSEg = sqrt( mean( ( df_test_regression$TARGET_LOSS_AMT - pg )^2 ) )

print( paste("TREE RMSE=", RMSEt ))
print( paste("RF RMSE=", RMSEr ))
print( paste("GB RMSE=", RMSEg ))

# The Random Forest model performed the best, with the lowest RMSE of approximately 4177.33
# The Random Forest model not only provides the most accurate predictions but also offers robustness against overfitting, making it suitable for operational use. While Gradient Boosting also provides reasonable accuracy, it doesn't outperform Random Forest. The Decision Tree, while the easiest to interpret, offers less accuracy. Thus, in this scenario, the slight loss of interpretability with Random Forest is outweighed by its superior performance, making it the recommended choice.

# Step 4: Probability / Severity Model Decision Tree
set.seed(3) # Ensuring reproducibility

# Split the full data into training and testing sets
FLAG <- sample(c(TRUE, FALSE), nrow(df), replace = TRUE, prob = c(0.7, 0.3))
df_train <- df[FLAG, ]
df_test <- df[!FLAG, ]

# Exclude TARGET_LOSS_AMT for the classification model training
df_train_excluded <- df_train
df_train_excluded$TARGET_LOSS_AMT <- NULL

# Filter the datasets to include only the instances with a default
df_train_defaults <- df_train[df_train$TARGET_BAD_FLAG == 1, ]
df_test_defaults <- df_test[df_test$TARGET_BAD_FLAG == 1, ]

# Re-train the Classification tree using Random Forest for TARGET_BAD_FLAG (probability of default)
rf_model_2 = randomForest(TARGET_BAD_FLAG ~ ., data = df_train_excluded, ntree=500, importance=TRUE )
importance( rf_model_2 )

pr_default = predict( rf_model_2, df_test)

# TREE

tr_model = rpart( data=df_train_defaults, TARGET_LOSS_AMT ~ ., control=control_parameters, method="poisson" )
rpart.plot( tr_model )
rpart.plot( tr_model, digits=-3, extra=100 )
tr_model$variable.importance

pt = predict( tr_model, df_test_defaults )
head( pt )
RMSEt = sqrt( mean( ( df_test_defaults$TARGET_LOSS_AMT - pt )^2 ) )

# RF

rf_model = randomForest( data=df_train_defaults, TARGET_LOSS_AMT ~ ., ntree=500, importance=TRUE )
importance( rf_model )
varImpPlot( rf_model )

pr = predict( rf_model, df_test_defaults)
head( pr )
RMSEr = sqrt( mean( ( df_test_defaults$TARGET_LOSS_AMT - pr )^2 ) )

# GRADIENT BOOSTING

gb_model = gbm( data=df_train_defaults, TARGET_LOSS_AMT ~ ., n.trees=500, distribution="poisson" )
summary.gbm( gb_model, cBars=10 )

pg = predict( gb_model, df_test_defaults, type="response" )
head( pg )
RMSEg = sqrt( mean( ( df_test_defaults$TARGET_LOSS_AMT - pg )^2 ) )

print( paste("TREE RMSE=", RMSEt ))
print( paste("RF RMSE=", RMSEr ))
print( paste("GB RMSE=", RMSEg ))

# Using the Gradient Boosting model predictions
selected_loss_model_predictions = pg  # Predictions from Gradient Boosting

# Ensure alignment in indices for calculating expected losses
# Get the probabilities of default for the records that are actually defaulted
pr_default_filtered = pr_default[df_test$TARGET_BAD_FLAG == 1]

# Calculate Expected Loss - multiply the predicted probability of default by the predicted loss
expected_loss = pr_default_filtered * selected_loss_model_predictions

# Calculate Actual Loss - from the test set, only for default records
actual_loss = df_test$TARGET_LOSS_AMT[df_test$TARGET_BAD_FLAG == 1]

# Calculate Overall RMSE for the Probability/Severity Model
overall_rmse = sqrt(mean((actual_loss - expected_loss)^2))
print(paste("Overall RMSE for Probability / Severity Model:", overall_rmse))

# The comparison between the models from Steps 3 and 4 centers on their applicability and complexity. Step 3 likely involved separate models for predicting default probability and loss severity, evaluated independently on metrics like AUC and RMSE, offering straightforward results but lacking integrated financial impact analysis. In contrast, Step 4 introduces a more sophisticated model that combines both probability of default and severity of loss, directly estimating expected losses which is crucial for comprehensive financial risk management. This model from Step 4 is particularly valuable in scenarios where understanding both the likelihood and potential impact of defaults is essential for making informed decisions about risk mitigation and capital allocation. Although more complex and possibly less interpretable, the integrated approach of Step 4 is recommended for its holistic view on financial outcomes, making it better suited for detailed risk assessment and decision support in financial contexts.

### Week 6 ###
# Step 1

df <- HMEQ_Scrubbed

# Step 2: Classification Models

set.seed(3) # Ensures reproducibility

# Exclude TARGET_LOSS_AMT from the predictors
df_FLAG <- df
df_FLAG$TARGET_LOSS_AMT <- NULL

# Splitting criteria
FLAG_FLAG <- sample(c(TRUE, FALSE), nrow(df_FLAG), replace = TRUE, prob = c(0.7, 0.3))
df_train_FLAG <- df_FLAG[FLAG_FLAG, ]
df_test_FLAG <- df_FLAG[!FLAG_FLAG, ]

# TREE
control_parameters <- rpart.control(maxdepth = 10)
tree_model_FLAG <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_FLAG, method = "class", control = control_parameters)
rpart.plot(tree_model_FLAG)
print(tree_model_FLAG$variable.importance)

pt_FLAG = predict( tree_model_FLAG, df_test_FLAG, type="prob" )
head( pt_FLAG )
pt2_FLAG = prediction( pt_FLAG[,2], df_test_FLAG$TARGET_BAD_FLAG)
pt3_FLAG = performance( pt2_FLAG, "tpr", "fpr" )

# RF
rf_model_FLAG = randomForest(TARGET_BAD_FLAG ~ ., data = df_train_FLAG, ntree=500, importance=TRUE )
importance( rf_model_FLAG )
varImpPlot( rf_model_FLAG )

pr_FLAG = predict( rf_model_FLAG, df_test_FLAG )
head( pr_FLAG )
pr2_FLAG = prediction( pr_FLAG, df_test_FLAG$TARGET_BAD_FLAG)
pr3_FLAG = performance( pr2_FLAG, "tpr", "fpr" )

# GRADIENT BOOSTING

gb_model_FLAG = gbm(TARGET_BAD_FLAG ~ ., data = df_train_FLAG, n.trees=500, distribution="bernoulli" )
summary.gbm( gb_model_FLAG, cBars=10 )

pg_FLAG = predict( gb_model_FLAG, df_test_FLAG, type="response" )
head( pg_FLAG )
pg2_FLAG = prediction( pg_FLAG, df_test_FLAG$TARGET_BAD_FLAG)
pg3_FLAG = performance( pg2_FLAG, "tpr", "fpr" )

# LOGISTIC REGRESSION using ALL the variables and FORWARD VARIABLE SELECTION

theUpper_LR_FLAG = glm( TARGET_BAD_FLAG ~ ., family = "binomial", data=df_train_FLAG )
theLower_LR_FLAG = glm( TARGET_BAD_FLAG ~ 1, family = "binomial", data=df_train_FLAG )

summary( theUpper_LR_FLAG )
summary( theLower_LR_FLAG )

lr_model_FLAG = stepAIC(theLower_LR_FLAG, direction="forward", scope=list(lower=theLower_LR_FLAG, upper=theUpper_LR_FLAG))
summary( lr_model_FLAG )

plr_FLAG = predict( lr_model_FLAG, df_test_FLAG, type="response" )
plr2_FLAG = prediction( plr_FLAG, df_test_FLAG$TARGET_BAD_FLAG )
plr3_FLAG = performance( plr2_FLAG, "tpr", "fpr" )

plot( plr3_FLAG, col="gold" )
abline(0,1,lty=2)
legend("bottomright",c("LOGISTIC REGRESSION FWD"),col=c("gold"), bty="y", lty=1 )

# LR STEP TREE
treeVars_FLAG = tree_model_FLAG$variable.importance
treeVars_FLAG = names(treeVars_FLAG)
treeVarsPlus_FLAG = paste( treeVars_FLAG, collapse="+")
F = as.formula( paste( "TARGET_BAD_FLAG ~", treeVarsPlus_FLAG ))

tree_LR_FLAG = glm( F, family = "binomial", data=df_train_FLAG )
theLower_LR_FLAG = glm( TARGET_BAD_FLAG ~ 1, family = "binomial", data=df_train_FLAG )

summary( tree_LR_FLAG )
summary( theLower_LR_FLAG )

lrt_model_FLAG = stepAIC(theLower_LR_FLAG, direction="both", scope=list(lower=theLower_LR_FLAG, upper=tree_LR_FLAG))
summary( lrt_model_FLAG )

plrt_FLAG = predict( lrt_model_FLAG, df_test_FLAG, type="response" )
plrt2_FLAG = prediction( plrt_FLAG, df_test_FLAG$TARGET_BAD_FLAG )
plrt3_FLAG = performance( plrt2_FLAG, "tpr", "fpr" )

plot( plrt3_FLAG, col="gray" )
abline(0,1,lty=2)
legend("bottomright",c("LOGISTIC REGRESSION TREE"),col=c("gray"), bty="y", lty=1 )

plot( pt3_FLAG, col="green" )
plot( pr3_FLAG, col="red", add=TRUE )
plot( pg3_FLAG, col="blue", add=TRUE )
plot( plr3_FLAG, col="gold", add=TRUE ) 
plot( plrt3_FLAG, col="gray", add=TRUE ) 

abline(0,1,lty=2)
legend("bottomright",c("TREE","RANDOM FOREST", "GRADIENT BOOSTING", "LOGIT REG FWD", "LOGIT REG TREE"),col=c("green","red","blue","gold","gray"), bty="y", lty=1 )

aucT = performance( pt2_FLAG, "auc" )@y.values
aucR = performance( pr2_FLAG, "auc" )@y.values
aucG = performance( pg2_FLAG, "auc" )@y.values
aucLR = performance( plr2_FLAG, "auc")@y.values
aucLRT = performance( plrt2_FLAG, "auc")@y.values

print( paste("TREE AUC=", aucT) )
print( paste("RF AUC=", aucR) )
print( paste("GB AUC=", aucG) )
print( paste("LR AUC=", aucLR) )
print( paste("LRT AUC=", aucLRT) )

# The Random Forest model performed best due to its highest AUC value, indicating superior ability to differentiate between the classes. Despite being less interpretable and more computationally demanding, I would recommend it for scenarios where accuracy is the top priority. If ease of interpretation or computational resources were major concerns, a simpler model like logistic regression could be considered.

# Step 3: Linear Regression

set.seed(3) # Ensures reproducibility

# Exclude TARGET_BAD_FLAG from the predictors for the regression tasks
df_AMT <- df
df_AMT$TARGET_BAD_FLAG <- NULL

# Splitting criteria
FLAG_AMT <- sample(c(TRUE, FALSE), nrow(df_AMT), replace = TRUE, prob = c(0.7, 0.3))
df_train_AMT <- df_AMT[FLAG_AMT, ]
df_test_AMT <- df_AMT[!FLAG_AMT, ]

# TREE

tr_model_AMT = rpart( data=df_train_AMT, TARGET_LOSS_AMT ~ ., control=control_parameters, method="poisson" )
rpart.plot( tr_model_AMT )
rpart.plot( tr_model_AMT, digits=-3, extra=100 )
tr_model_AMT$variable.importance

pt_AMT = predict( tr_model_AMT, df_test_AMT )
head( pt_AMT )
RMSEt = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - pt_AMT )^2 ) )

# RF

rf_model_AMT = randomForest( data=df_train_AMT, TARGET_LOSS_AMT ~ ., ntree=500, importance=TRUE )
importance( rf_model_AMT )
varImpPlot( rf_model_AMT )

pr_AMT = predict( rf_model_AMT, df_test_AMT )
head( pr_AMT )
RMSEr = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - pr_AMT )^2 ) )

# GRADIENT BOOSTING

gb_model_AMT = gbm( data=df_train_AMT, TARGET_LOSS_AMT ~ ., n.trees=500, distribution="poisson" )
summary.gbm( gb_model_AMT, cBars=10 )

pg_AMT = predict( gb_model_AMT, df_test_AMT, type="response" )
head( pg_AMT )
RMSEg = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - pg_AMT )^2 ) )

# LINEAR REGRESSION using ALL the variable and BACKWARD VARIABLE SELECTION

theUpper_LR_AMT = lm( TARGET_LOSS_AMT ~ ., data=df_train_AMT )
theLower_LR_AMT = lm( TARGET_LOSS_AMT ~ 1, data=df_train_AMT )

summary( theUpper_LR_AMT )
summary( theLower_LR_AMT )

lr_model_AMT = stepAIC(theUpper_LR_AMT, direction="backward", scope=list(lower=theLower_LR_AMT, upper=theUpper_LR_AMT))
summary( lr_model_AMT )

plr_AMT = predict( lr_model_AMT, df_test_AMT )
head( plr_AMT )
RMSElr = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - plr_AMT )^2 ) )

# LR STEP TREE
treeVars_AMT = tr_model_AMT$variable.importance
treeVars_AMT = names(treeVars_AMT)
treeVarsPlus_AMT = paste( treeVars_AMT, collapse="+")
F = as.formula( paste( "TARGET_LOSS_AMT ~", treeVarsPlus_AMT ))

tree_LR_AMT = lm( F, data=df_train_AMT )
theLower_LR_AMT = lm( TARGET_LOSS_AMT ~ 1, data=df_train_AMT )

summary( tree_LR_AMT )
summary( theLower_LR_AMT )

lrt_model_AMT = stepAIC(theLower_LR_AMT, direction="both", scope=list(lower=theLower_LR_AMT, upper=tree_LR_AMT))
summary( lrt_model_AMT )


plr_tree_AMT = predict( tree_LR_AMT, df_test_AMT )
head( plr_tree_AMT )
RMSElr_tree = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - plr_tree_AMT )^2 ) )

plr_tree_step_AMT = predict( lrt_model_AMT, df_test_AMT )
head( plr_tree_step_AMT )
RMSElr_tree_step = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - plr_tree_step_AMT )^2 ) )

print( paste("TREE RMSE=", RMSEt ))
print( paste("RF RMSE=", RMSEr ))
print( paste("GB RMSE=", RMSEg ))

print( paste("LR BACK RMSE=",  RMSElr ))
print( paste("LR TREE RMSE=",  RMSElr_tree ))
print( paste("LR TREE STEP RMSE=", RMSElr_tree_step ))

# The Random Forest model had the lowest RMSE, indicating it was the most accurate for this dataset. I'd recommend it for scenarios where accuracy is crucial. If you need a faster or more interpretable model, Linear Regression could be a simpler but less accurate alternative.

# Step 4: Probability / Severity Model
set.seed(3) # Ensuring reproducibility

# Split the full data into training and testing sets
FLAG <- sample(c(TRUE, FALSE), nrow(df), replace = TRUE, prob = c(0.7, 0.3))
df_train <- df[FLAG, ]
df_test <- df[!FLAG, ]

# Exclude TARGET_LOSS_AMT for the classification model training
df_train_excluded <- df_train
df_train_excluded$TARGET_LOSS_AMT <- NULL

# Filter the datasets to include only the instances with a default
df_train_defaults <- df_train[df_train$TARGET_BAD_FLAG == 1, ]
df_test_defaults <- df_test[df_test$TARGET_BAD_FLAG == 1, ]

# Logistic Regression with Forward Selection
theUpper_LR = glm(TARGET_BAD_FLAG ~ ., family = "binomial", data = df_train_excluded)
theLower_LR = glm(TARGET_BAD_FLAG ~ 1, family = "binomial", data = df_train_excluded)
logit_model = stepAIC(theLower_LR, direction = "forward", scope = list(lower = theLower_LR, upper = theUpper_LR))
summary(logit_model)

# Linear Regression with Backward Selection
theUpper_LM = lm(TARGET_LOSS_AMT ~ ., data = df_train_defaults)
theLower_LM = lm(TARGET_LOSS_AMT ~ 1, data = df_train_defaults)
lm_model = stepAIC(theUpper_LM, direction = "backward", scope = list(lower = theLower_LM, upper = theUpper_LM))
summary(lm_model)

# Predictions
prob_default = predict(logit_model, df_test, type = "response")
loss_given_default = predict(lm_model, df_test_defaults, type = "response")

# Calculate Expected Loss
expected_loss = df_test$TARGET_BAD_FLAG * prob_default * loss_given_default

# Actual Loss - assume TARGET_LOSS_AMT is 0 where TARGET_BAD_FLAG is 0
actual_loss = ifelse(df_test$TARGET_BAD_FLAG == 1, df_test$TARGET_LOSS_AMT, 0)

# RMSE Calculation
RMSE = sqrt(mean((actual_loss - expected_loss)^2))
print(paste("RMSE for Probability / Severity Model =", RMSE))

# I would recommend using the Random Forest model from Step 3 over the Probability / Severity Model from Step 4 for applications where prediction accuracy is paramount. However, if the context requires assessing not only the likelihood but also the severity of outcomes, adjustments or enhancements to the Probability / Severity Model might be necessary to improve its predictive performance.

### Week 7 ###
# Step 1

df <- HMEQ_Scrubbed

SEED = 123
set.seed( SEED )

# Step 2: PCA Analysis

df_pca = df
df_pca$TARGET_BAD_FLAG = NULL
df_pca$TARGET_LOSS_AMT = NULL
pca2 = prcomp(df_pca[,c(1,2,4,6,8,10,12,14,16,18)] ,center=TRUE, scale=TRUE)
summary(pca2)
plot(pca2, type = "l")
df_new <- data.frame(predict(pca2, df_pca))
df_new$TARGET_BAD_FLAG <- df$TARGET_BAD_FLAG

# Print the weights of the Principal Components
print(pca2$rotation)

# Plot the first two principal components colored by the Target Flag
ggplot(df_new, aes(x = PC1, y = PC2, color = factor(TARGET_BAD_FLAG))) +
  geom_point() +
  scale_color_manual(values = c("blue", "red"), 
                     labels = c("Non Default", "Default")) +
  theme_minimal() +
  ggtitle("Scatter Plot of the First Two Principal Components") +
  xlab("Principal Component 1") +
  ylab("Principal Component 2")

# The first two Principal Components do not clearly separate the default and non-default classes, as there is significant overlap between them.

# Step 3: tSNE Analysis

dfu = df
dfu$TARGET_LOSS_AMT = NULL
dfu = unique(dfu)

# Conduct tSNE analysis with Perplexity = 30
theTSNE = Rtsne(dfu[, c(2,3,5,7,9,11,13,15,17,19)], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
dfu$TS1 = theTSNE$Y[,1]
dfu$TS2 = theTSNE$Y[,2]

# Plotting the results with TARGET_BAD_FLAG
library(ggplot2)
ggplot(dfu, aes(x = TS1, y = TS2, color = factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.5) +
  labs(color = "Target Flag") +
  ggtitle("tSNE Plot with Perplexity = 30")

# The tSNE plot with a perplexity of 30 shows considerable overlap between defaults and non-defaults, indicating that the tSNE values at this perplexity level are not highly predictive of the Target Flag.

# Conduct tSNE with a higher perplexity, e.g., 50
theTSNE_high = Rtsne(dfu[, c(2,3,5,7,9,11,13,15,17,19)], dims = 2, perplexity=50, verbose=TRUE, max_iter = 500)
dfu$TS1_high = theTSNE_high$Y[,1]
dfu$TS2_high = theTSNE_high$Y[,2]
ggplot(dfu, aes(x = TS1_high, y = TS2_high, color = factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.5) +
  labs(color = "Target Flag") +
  ggtitle("tSNE Plot with Perplexity = 50")

# Conduct tSNE with a lower perplexity, e.g., 10
theTSNE_low = Rtsne(dfu[, c(2,3,5,7,9,11,13,15,17,19)], dims = 2, perplexity=10, verbose=TRUE, max_iter = 500)
dfu$TS1_low = theTSNE_low$Y[,1]
dfu$TS2_low = theTSNE_low$Y[,2]
ggplot(dfu, aes(x = TS1_low, y = TS2_low, color = factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.5) +
  labs(color = "Target Flag") +
  ggtitle("tSNE Plot with Perplexity = 10")

# The plot with perplexity 50 shows a tendency towards more discernible clustering compared to the others, which may suggest a marginally better predictive ability for the Target Flag.

# Train Random Forest Models
P = paste(colnames(dfu)[c(2,3,5,7,9,11,13,15,17,19)], collapse = "+")
F1 = as.formula( paste("TS1 ~", P ) )
F2 = as.formula( paste("TS2 ~", P ) )
print( F1 )
print( F2 )
ts1_model_rf = randomForest( data=dfu, F1, ntree=500, importance=TRUE )
ts2_model_rf = randomForest( data=dfu, F2, ntree=500, importance=TRUE )

# Step 4: Tree and Regression Analysis on the Original Data

df_model = df
df_model$TARGET_LOSS_AMT = NULL

head( df_model )

# Decision Tree
tr_set = rpart.control( maxdepth = 10 )
t1G = rpart( data=df_model, TARGET_BAD_FLAG ~ ., control=tr_set, method="class", parms=list(split='gini') )
t1E = rpart( data=df_model, TARGET_BAD_FLAG ~ ., control=tr_set, method="class", parms=list(split='information') )

rpart.plot( t1G )
rpart.plot( t1E )

t1G$variable.importance
t1E$variable.importance

# In both decision tree models (t1G and t1E), M_DEBTINC appears to be the most important predictor, followed by IMP_DEBTINC, IMP_DELINQ, and M_VALUE. 

# Logistic Regression
theUpper_LR = glm( TARGET_BAD_FLAG ~ ., family = "binomial", data=df_model )
theLower_LR = glm( TARGET_BAD_FLAG ~ 1, family = "binomial", data=df_model )

summary( theUpper_LR )
summary( theLower_LR )

lr_model = stepAIC(theLower_LR, direction="forward", scope=list(lower=theLower_LR, upper=theUpper_LR))
summary( lr_model )

# In the logistic regression model (lr_model), several variables have statistically significant coefficients: M_DEBTINC, IMP_DELINQ, IMP_DEBTINC, M_VALUE, M_DEROG, IMP_DEROG, IMP_NINQ, FLAG.Job.Office, M_YOJ, FLAG.Job.Sales, M_DELINQ, M_CLNO, IMP_CLNO, FLAG.Job.Other, IMP_VALUE, IMP_YOJ, FLAG.Job.Self, FLAG.Job.Mgr, FLAG.Job.ProfExe, M_MORTDUE, and IMP_MORTDUE.

pG = predict( t1G, df_model )
pG2 = prediction( pG[,2], df_model$TARGET_BAD_FLAG )
pG3 = performance( pG2, "tpr", "fpr" )

pE = predict( t1E, df_model )
pE2 = prediction( pE[,2], df_model$TARGET_BAD_FLAG )
pE3 = performance( pE2, "tpr", "fpr" )

plr = predict( lr_model, df_model, type="response" )
plr2 = prediction( plr, df_model$TARGET_BAD_FLAG )
plr3 = performance( plr2, "tpr", "fpr" )

plot( pG3, col="red" )
plot( pE3, col="green", add=TRUE )
plot( plr3, col="blue", add=TRUE )
abline(0,1,lty=2)
legend("bottomright",c("GINI","ENTROPY","REGRESSION"),col=c("red","green","blue"), bty="y", lty=1 )

aucG = performance( pG2, "auc" )@y.values
aucE = performance( pE2, "auc" )@y.values
aucR = performance( plr2, "auc" )@y.values

print( aucG )
print( aucE )
print( aucR )

# Step 5: Tree and Regression Analysis on the PCA/tSNE Data

df_model = df
df_model$TARGET_LOSS_AMT = NULL

df_model$PC1 = df_new[,"PC1"]
df_model$PC2 = df_new[,"PC2"]
df_model$PC3 = df_new[,"PC3"]
df_model$PC4 = df_new[,"PC4"]

df_model$TS1M_RF = predict( ts1_model_rf, df_model )
df_model$TS2M_RF = predict( ts2_model_rf, df_model )

df_model$LOAN = NULL
df_model$IMP_MORTDUE = NULL
df_model$IMP_VALUE = NULL
df_model$IMP_YOJ = NULL
df_model$IMP_DEROG = NULL
df_model$IMP_DELINQ = NULL
df_model$IMP_CLAGE = NULL
df_model$IMP_NINQ = NULL
df_model$IMP_CLNO = NULL
df_model$IMP_DEBTINC = NULL

head( df_model )

# Decision Tree
tr_set = rpart.control( maxdepth = 10 )
t1G = rpart( data=df_model, TARGET_BAD_FLAG ~ ., control=tr_set, method="class", parms=list(split='gini') )
t1E = rpart( data=df_model, TARGET_BAD_FLAG ~ ., control=tr_set, method="class", parms=list(split='information') )

rpart.plot( t1G )
rpart.plot( t1E )

t1G$variable.importance
t1E$variable.importance

# The models incorporate key variables like M_DEBTINC, M_VALUE, and M_CLAGE, along with principal components (PC1, PC2, PC3, PC4) and t-SNE derived features (TS1M_RF, TS2M_RF). These advanced statistical techniques—PCA and t-SNE—help capture complex, multidimensional patterns in the data, enhancing the models' ability to predict outcomes effectively. Notably, M_DEBTINC and PC2 emerge as particularly significant predictors in both models. 

# Logistic Rregression
theUpper_LR = glm( TARGET_BAD_FLAG ~ ., family = "binomial", data=df_model )
theLower_LR = glm( TARGET_BAD_FLAG ~ 1, family = "binomial", data=df_model )

summary( theUpper_LR )
summary( theLower_LR )

lr_model = stepAIC(theLower_LR, direction="forward", scope=list(lower=theLower_LR, upper=theUpper_LR))
summary( lr_model )

# The logistic regression model incorporates a diverse set of variables including both traditional loan and demographic data (like M_DEBTINC, M_VALUE, M_DEROG, and job type flags) as well as advanced statistical features such as principal components (PC2) and t-SNE derived variables (TS2M_RF and TS1M_RF). The inclusion of PC2 suggests that the second principal component was particularly relevant in capturing variance in the dataset, possibly encapsulating multiple correlated variables into a single predictor. The t-SNE features (TS2M_RF and TS1M_RF), although included, show relatively lower importance in the model, indicating that while they contribute to the model, their impact is less dominant compared to direct financial measures or PCA features. This combination of variables suggests a robust approach to understanding the factors influencing loan default, leveraging both raw and transformed features to enhance predictive accuracy.

# ROC and AUC
pG = predict( t1G, df_model )
pG2 = prediction( pG[,2], df_model$TARGET_BAD_FLAG )
pG3 = performance( pG2, "tpr", "fpr" )

pE = predict( t1E, df_model )
pE2 = prediction( pE[,2], df_model$TARGET_BAD_FLAG )
pE3 = performance( pE2, "tpr", "fpr" )

plr = predict( lr_model, df_model, type="response" )
plr2 = prediction( plr, df_model$TARGET_BAD_FLAG )
plr3 = performance( plr2, "tpr", "fpr" )

plot( pG3, col="red" )
plot( pE3, col="green", add=TRUE )
plot( plr3, col="blue", add=TRUE )
abline(0,1,lty=2)
legend("bottomright",c("GINI","ENTROPY","REGRESSION"),col=c("red","green","blue"), bty="y", lty=1 )

aucG = performance( pG2, "auc" )@y.values
aucE = performance( pE2, "auc" )@y.values
aucR = performance( plr2, "auc" )@y.values

print( aucG )
print( aucE )
print( aucR )

# The PCA and t-SNE values have improved the models' abilities to differentiate between the classes effectively, especially in logistic regression. This suggests that these dimensionality reduction techniques are vital in handling complex datasets where direct relationships between variables and outcomes are not easily discernible. Thus, when compared to using the original dataset alone, incorporating PCA and t-SNE seems to provide a substantial improvement in model performance, making them valuable tools in predictive analytics, particularly for datasets with complex, high-dimensional structures.

### Week 8 ###
# Step 1

df <- HMEQ_Scrubbed  

SEED = 123
set.seed(SEED)

TARGET = "TARGET_BAD_FLAG"

# Step 2: PCA Analysis
df_pca = df
df_pca$TARGET_BAD_FLAG = NULL
df_pca$TARGET_LOSS_AMT = NULL
pca = prcomp(df_pca[,c(1,2,4,6,8,10,12,14,16,18)] ,center=TRUE, scale=TRUE)
summary(pca)
plot(pca, type = "l")

df_new <- data.frame(predict(pca, df_pca))
df_new$TARGET_BAD_FLAG <- df$TARGET_BAD_FLAG

# Print the weights of the Principal Components
print(pca$rotation)

# Plot the first two principal components
ggplot(df_new, aes(x = PC1, y = PC2)) +
  geom_point() +
  theme_minimal() +
  ggtitle("Scatter Plot of the First Two Principal Components") +
  xlab("Principal Component 1") +
  ylab("Principal Component 2")

# Step 3: Cluster Analysis - Find the Number of Clusters
df_new = data.frame(predict(pca, df_pca))
df_kmeans = df_new[1:2]
print(head(df_kmeans))
plot(df_kmeans$PC1, df_kmeans$PC2)

MAX_N = 10
WSS = numeric(MAX_N)
for (N in 1:MAX_N) {
  km = kmeans(df_kmeans, centers=N, nstart=20)
  WSS[N] = km$tot.withinss
}
df_wss = as.data.frame(WSS)
df_wss$clusters = 1:MAX_N

scree_plot = ggplot(df_wss, aes(x=clusters, y=WSS, group=1)) +
  geom_point(size=4) +
  geom_line() +
  scale_x_continuous(breaks=c(2,4,6,8,10)) +
  xlab("Number of Clusters")
scree_plot

# 3 clusters would likely be the most effective choice, balancing between too many clusters (which might overfit and capture noise rather than true structure) and too few (which might underfit and miss important patterns). 

# Step 4: Cluster Analysis
BEST_N = 5
km = kmeans(df_kmeans, centers=BEST_N, nstart=20)
print(km$size)
print(km$centers)

kf = as.kcca(object=km, data=df_kmeans, save.data=TRUE)
kfi = kcca2df(kf)
agg = aggregate(kfi$value, list(kfi$variable, kfi$group), FUN=mean)
barplot(kf)

clus = predict(kf, df_kmeans)
plot(df_kmeans$PC1, df_kmeans$PC2)
plot(df_kmeans$PC1, df_kmeans$PC2, col=clus)
legend(x="topright", legend=c(1:BEST_N), fill=c(1:BEST_N))

df$CLUSTER = clus
agg = aggregate(df$TARGET_BAD_FLAG, list(df$CLUSTER), FUN=mean)

# Step 5: Describe the Clusters Using Decision Trees

library(rpart)
library(rpart.plot)
df_tree = df_pca
df_tree$CLUSTER = as.factor(clus)
dt = rpart(CLUSTER ~ ., data=df_tree)
dt = rpart(CLUSTER ~ ., data=df_tree, maxdepth=3)
rpart.plot(dt)

# The decision tree effectively segments the dataset into five distinct clusters based on financial indicators such as property value, outstanding mortgage, derogatory marks, and recent credit inquiries. Cluster 1 represents individuals with past credit issues and low property values, suggesting financial recovery or stabilization. Cluster 2 indicates more credit-stable individuals with similar property values but without derogatory records. Cluster 3 includes those potentially underwater on their mortgages, implying financial distress. Cluster 4 consists of individuals with moderate mortgages but high credit-seeking behavior, possibly indicating financial strain or opportunity seizing. Finally, Cluster 5 represents a wealthier segment with higher property values, suggesting better financial health. These clusters seem logically organized, reflecting varying financial conditions and behaviors that can aid in targeted financial decision-making and risk assessment.

# Step 6: Comment

# In a corporate setting, the clusters derived from your analysis can be utilized to enhance targeted marketing efforts, optimize risk management strategies, and improve customer segmentation. By understanding the distinct characteristics of each cluster, companies can tailor their services and products to meet the specific needs and preferences of different customer groups, thereby increasing efficiency and effectiveness in their operations. This approach not only aids in precise marketing and risk mitigation but also supports strategic decision-making and the development of predictive models that anticipate customer behaviors.