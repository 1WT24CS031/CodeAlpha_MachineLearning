# Credit Scoring Model

A machine learning project that predicts whether a credit card client is likely to default on their payment in the following month.

This project was developed as part of the CodeAlpha Machine Learning Internship.

---

## 📌 Project Overview

Credit default prediction is a binary classification problem where the model predicts whether a customer will default on their next credit card payment.

The project uses financial and demographic information such as:

- Credit limit
- Gender
- Education
- Marital status
- Age
- Previous repayment status
- Bill amounts
- Previous payment amounts

Several machine learning models were trained and evaluated, followed by hyperparameter tuning of the Random Forest model.

---

## 🎯 Objective

The main objectives of this project are to:

1. Load and clean the credit card dataset.
2. Perform exploratory data analysis (EDA).
3. Engineer additional financial features.
4. Train multiple classification models.
5. Evaluate models using classification metrics.
6. Tune the best-performing model.
7. Save the final trained model and scaler.
8. Generate visualizations for model evaluation.

---

## 📊 Dataset

The dataset used is the **Default of Credit Card Clients Dataset** from the UCI Machine Learning Repository.

Dataset source:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset contains:

- 30,000 customer records
- 23 predictive features
- A binary target variable

### Target Variable

`default payment next month`

Target values:

- `0` → No default
- `1` → Default

The dataset contains an imbalanced target distribution, with approximately 78% non-default cases and 22% default cases.

Because of this imbalance, metrics such as Precision, Recall and F1-Score were considered along with Accuracy.

---

## 🔍 Exploratory Data Analysis

The project includes visualizations for:

- Distribution of default vs non-default customers
- Credit limit distribution by default status

The generated plots are stored in the `outputs` folder.

---

## ⚙️ Feature Engineering

Additional features were created to capture overall customer financial behavior.

### Average Bill Amount

Average of the six monthly bill amounts.

`AVG_BILL_AMT`

### Total Bill Amount

Sum of the six monthly bill amounts.

`TOTAL_BILL_AMT`

### Average Payment Amount

Average of the six monthly payment amounts.

`AVG_PAY_AMT`

### Total Payment Amount

Sum of the six monthly payment amounts.

`TOTAL_PAY_AMT`

### Average Payment Delay

Average repayment status across the available repayment history.

`AVG_PAYMENT_DELAY`

### Maximum Payment Delay

Maximum repayment status recorded for the customer.

`MAX_PAYMENT_DELAY`

---

## 🤖 Machine Learning Models

Three classification algorithms were evaluated:

### 1. Logistic Regression

A linear classification algorithm used as a baseline model.

### 2. Decision Tree

A tree-based classification algorithm capable of learning nonlinear decision rules.

### 3. Random Forest

An ensemble of decision trees that generally provides stronger performance and better generalization than a single decision tree.

---

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

F1-Score was given particular importance because the dataset contains fewer default cases than non-default cases.

---

## 🔧 Hyperparameter Tuning

RandomizedSearchCV was used to tune the Random Forest model.

The search considered parameters including:

- Number of estimators
- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples required at a leaf
- Maximum features considered for splitting

The tuning process used:

- 10 randomized parameter combinations
- 3-fold cross-validation
- F1-Score as the optimization metric

### Best Parameters

```text
n_estimators = 200
max_depth = 10
min_samples_split = 5
min_samples_leaf = 1
max_features = sqrt