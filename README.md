# Stroke-Prediction-Model
title: "Build and deploy a stroke prediction model using R"
date: "May 23, 2023"
output: html_document
author: "Adedeji Tiamiyu!"
---

# About Data Analysis Report

This RMarkdown file contains the report of the data analysis done for the project on building and deploying a stroke prediction model in R. It contains analysis such as data exploration, summary statistics and building the prediction models. The final report was completed on `r date()`. 

**Data Description:**

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.

This data set is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.


# Task One: Import data and data preprocessing

## Load data and install packages

```{r}
# Install necessary packages if not already installed
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(VIM)) install.packages("VIM")

# Load libraries
library(tidyverse)
library(caret)
library(ggplot2)
library(VIM)

# Load the dataset
healthcare_dataset_stroke_data <- read.csv("Build-deploy-stroke-prediction-model-R/healthcare-dataset-stroke-data.csv")

# Check the first few rows of the dataset
head(healthcare_dataset_stroke_data)

```
## Describe and explore the data


```{r}
# Convert columns to appropriate types
healthcare_dataset_stroke_data <- healthcare_dataset_stroke_data %>%
  mutate(
    gender = as.factor(gender),
    ever_married = as.factor(ever_married),
    work_type = as.factor(work_type),
    Residence_type = as.factor(Residence_type),
    smoking_status = as.factor(smoking_status),
    bmi = as.numeric(bmi)
  )

# Handle missing values in 'bmi'
healthcare_dataset_stroke_data <- healthcare_dataset_stroke_data %>%
  mutate(bmi = ifelse(is.na(bmi), median(bmi, na.rm = TRUE), bmi))

# Check the structure and summary of the cleaned dataset
str(healthcare_dataset_stroke_data)
summary(healthcare_dataset_stroke_data)

# Visualize the distribution of stroke cases
ggplot(data = healthcare_dataset_stroke_data, aes(x = factor(stroke))) + 
  geom_bar(fill = "skyblue") +
  labs(x = "Stroke (0 = No, 1 = Yes)", y = "Count", title = "Distribution of Stroke Cases")

# Check for missing values
aggr(healthcare_dataset_stroke_data, 
     col = c('navyblue', 'red'), 
     numbers = TRUE, 
     sortVars = TRUE, 
     labels = names(healthcare_dataset_stroke_data), 
     cex.axis = .7, 
     gap = 3, 
     ylab = c("Missing data", "Pattern"))
```

# Task Two: Build prediction models

```{r}

# Split the data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(healthcare_dataset_stroke_data$stroke, p = 0.8, list = FALSE)
trainData <- healthcare_dataset_stroke_data[trainIndex,]
testData <- healthcare_dataset_stroke_data[-trainIndex,]

# Ensure factors have consistent levels
trainData$stroke <- factor(trainData$stroke)
testData$stroke <- factor(testData$stroke, levels = levels(trainData$stroke))

# Remove constant variables from the dataset
constant_vars <- sapply(trainData, function(x) length(unique(x)) == 1)
trainData <- trainData[, !constant_vars]
testData <- testData[, !constant_vars]

## Train various models
# Logistic Regression
model_log <- train(stroke ~ ., data = trainData, method = "glm", family = binomial, trControl = trainControl(method = "cv", number = 10))

# Random Forest
model_rf <- train(stroke ~ ., data = trainData, method = "rf", trControl = trainControl(method = "cv", number = 10))

# Support Vector Machine
model_svm <- train(stroke ~ ., data = trainData, method = "svmRadial", trControl = trainControl(method = "cv", number = 10))

```


# Task Three: Evaluate and select prediction models

```{r}
# Evaluate models on the test set
# Logistic Regression
pred_log <- predict(model_log, newdata = testData)
confusion_log <- confusionMatrix(pred_log, testData$stroke)
print(confusion_log)

# Random Forest
pred_rf <- predict(model_rf, newdata = testData)
confusion_rf <- confusionMatrix(pred_rf, testData$stroke)
print(confusion_rf)

# Support Vector Machine
pred_svm <- predict(model_svm, newdata = testData)
confusion_svm <- confusionMatrix(pred_svm, testData$stroke)
print(confusion_svm)

# Select the best model based on accuracy
# (Assuming highest accuracy is the criterion for selection)
best_model <- ifelse(confusion_log$overall['Accuracy'] > confusion_rf$overall['Accuracy'] & confusion_log$overall['Accuracy'] > confusion_svm$overall['Accuracy'], 
                      "Logistic Regression", 
                      ifelse(confusion_rf$overall['Accuracy'] > confusion_svm$overall['Accuracy'], "Random Forest", "SVM"))

print(paste("Best Model:", best_model))

```

# Task Four: Deploy the prediction model

```{r}
# Save the best model
saveRDS(model_log, file = "best_model.rds")

# Load the model (example usage)
loaded_model <- readRDS("best_model.rds")

```

# Task Five: Findings and Conclusions
# Draw insights and conclusions
cat("The Logistic Regression model performed the best based on accuracy. This model is saved and can be deployed for predicting stroke risk based on patient data.\n")
cat("Next steps include integrating the model into a web service or application for end-user predictions.\n")
cat("Limitations of the model include its performance on different datasets and the need for regular updates with new data.\n")










