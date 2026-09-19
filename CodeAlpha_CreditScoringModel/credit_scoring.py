from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

project_dir = Path(__file__).parent

file_path = project_dir / "default of credit card clients.xls"

output_dir = project_dir / "outputs"
model_dir = project_dir / "models"

output_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_excel(file_path)


# ============================================================
# 3. CLEAN DATASET
# ============================================================

df.columns = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)

df = df.apply(pd.to_numeric, errors="coerce")


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

target_column = "default payment next month"

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\nNumber of samples in each class:")
print(df[target_column].value_counts())

print("\nTarget percentage:")
print(
    (df[target_column].value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 7. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# Graph 1: Default Distribution

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x=target_column
)

plt.title("Credit Card Default Distribution")
plt.xlabel("Default Payment Next Month")
plt.ylabel("Number of Customers")

plt.xticks(
    [0, 1],
    ["No Default", "Default"]
)

plt.tight_layout()

graph1_path = output_dir / "01_default_distribution.png"

plt.savefig(
    graph1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)

print("\nGraph 1 displayed.")
print("Saved:", graph1_path)


# Graph 2: Credit Limit vs Default

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x=target_column,
    y="LIMIT_BAL"
)

plt.title("Credit Limit Distribution by Default Status")
plt.xlabel("Default Payment Next Month")
plt.ylabel("Credit Limit")

plt.xticks(
    [0, 1],
    ["No Default", "Default"]
)

plt.tight_layout()

graph2_path = output_dir / "02_credit_limit_vs_default.png"

plt.savefig(
    graph2_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)

print("Graph 2 displayed.")
print("Saved:", graph2_path)

plt.pause(3)

plt.close("all")


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


bill_columns = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]

payment_columns = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

payment_status_columns = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]


# Average and total bills

df["AVG_BILL_AMT"] = df[bill_columns].mean(axis=1)

df["TOTAL_BILL_AMT"] = df[bill_columns].sum(axis=1)


# Average and total payments

df["AVG_PAY_AMT"] = df[payment_columns].mean(axis=1)

df["TOTAL_PAY_AMT"] = df[payment_columns].sum(axis=1)


# Repayment behavior

df["AVG_PAYMENT_DELAY"] = df[payment_status_columns].mean(axis=1)

df["MAX_PAYMENT_DELAY"] = df[payment_status_columns].max(axis=1)


new_features = [
    "AVG_BILL_AMT",
    "TOTAL_BILL_AMT",
    "AVG_PAY_AMT",
    "TOTAL_PAY_AMT",
    "AVG_PAYMENT_DELAY",
    "MAX_PAYMENT_DELAY"
]

print("\nNew features created:")

for feature in new_features:
    print("-", feature)

print("\nSample of engineered features:")
print(df[new_features].head())

print("\nNew dataset shape after feature engineering:")
print(df.shape)


# ============================================================
# 9. REMOVE MISSING VALUES
# ============================================================

df = df.dropna()


# ============================================================
# 10. FEATURE / TARGET SEPARATION
# ============================================================

X = df.drop(
    columns=[target_column, "ID"]
)

y = df[target_column]


print("\n" + "=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

print("\nFeature shape:")
print(X.shape)

print("\nFeatures:")
print(X.columns.tolist())


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 12. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 13. BASELINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 14. TRAIN BASELINE MODELS
# ============================================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


# ============================================================
# 15. BASELINE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).T

print("\n" + "=" * 60)
print("BASELINE MODEL COMPARISON")
print("=" * 60)

print(
    results_df.round(4)
)


# ============================================================
# 16. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

print("\nSearching for the best Random Forest parameters...")
print("This may take a few minutes.")


rf_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


param_distributions = {

    "n_estimators": [
        100,
        200,
        300,
        400
    ],

    "max_depth": [
        None,
        10,
        20,
        30
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        "log2"
    ]
}


random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="f1",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


random_search.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 17. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("BEST RANDOM FOREST PARAMETERS")
print("=" * 60)

print(
    random_search.best_params_
)

print(
    "\nBest cross-validation F1-Score:",
    round(
        random_search.best_score_,
        4
    )
)


# ============================================================
# 18. FINAL TUNED RANDOM FOREST
# ============================================================

best_rf = random_search.best_estimator_


y_pred_tuned = best_rf.predict(
    X_test_scaled
)

y_prob_tuned = best_rf.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 19. FINAL TUNED MODEL EVALUATION
# ============================================================

tuned_accuracy = accuracy_score(
    y_test,
    y_pred_tuned
)

tuned_precision = precision_score(
    y_test,
    y_pred_tuned
)

tuned_recall = recall_score(
    y_test,
    y_pred_tuned
)

tuned_f1 = f1_score(
    y_test,
    y_pred_tuned
)

tuned_roc_auc = roc_auc_score(
    y_test,
    y_prob_tuned
)


print("\n" + "=" * 60)
print("TUNED RANDOM FOREST RESULTS")
print("=" * 60)

print(
    "\nAccuracy :",
    round(tuned_accuracy, 4)
)

print(
    "Precision:",
    round(tuned_precision, 4)
)

print(
    "Recall   :",
    round(tuned_recall, 4)
)

print(
    "F1-Score :",
    round(tuned_f1, 4)
)

print(
    "ROC-AUC  :",
    round(tuned_roc_auc, 4)
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_tuned
    )
)


print("Confusion Matrix:")

final_cm = confusion_matrix(
    y_test,
    y_pred_tuned
)

print(final_cm)


# ============================================================
# 20. BASELINE VS TUNED RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("BASELINE VS TUNED RANDOM FOREST")
print("=" * 60)

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],

    "Baseline Random Forest": [
        results_df.loc[
            "Random Forest",
            "Accuracy"
        ],

        results_df.loc[
            "Random Forest",
            "Precision"
        ],

        results_df.loc[
            "Random Forest",
            "Recall"
        ],

        results_df.loc[
            "Random Forest",
            "F1-Score"
        ],

        results_df.loc[
            "Random Forest",
            "ROC-AUC"
        ]
    ],

    "Tuned Random Forest": [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1,
        tuned_roc_auc
    ]

})


print(
    comparison.round(4)
)


# ============================================================
# 21. FINAL CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("FINAL CONFUSION MATRIX")
print("=" * 60)


plt.figure(figsize=(7, 5))

sns.heatmap(
    final_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Default", "Default"],
    yticklabels=["No Default", "Default"]
)

plt.title("Confusion Matrix - Tuned Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

confusion_matrix_path = (
    output_dir / "03_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)

print(
    "Confusion matrix saved:",
    confusion_matrix_path
)

plt.pause(3)
plt.close()


# ============================================================
# 22. ROC CURVE
# ============================================================

print("\n" + "=" * 60)
print("ROC CURVE")
print("=" * 60)


fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob_tuned
)


plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {tuned_roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Tuned Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()

roc_curve_path = output_dir / "04_roc_curve.png"

plt.savefig(
    roc_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)

print(
    "ROC curve saved:",
    roc_curve_path
)

plt.pause(3)
plt.close()


# ============================================================
# 23. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": best_rf.feature_importances_

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 10 important features:")

print(
    feature_importance.head(10).round(4)
)


plt.figure(figsize=(9, 7))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Feature Importances - Tuned Random Forest"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

feature_importance_path = (
    output_dir / "05_feature_importance.png"
)

plt.savefig(
    feature_importance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)

print(
    "Feature importance saved:",
    feature_importance_path
)

plt.pause(3)
plt.close()


# ============================================================
# 24. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING FINAL MODEL")
print("=" * 60)


model_path = model_dir / "credit_scoring_model.pkl"

scaler_path = model_dir / "credit_scoring_scaler.pkl"


joblib.dump(
    best_rf,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)


print(
    "\nFinal model saved:",
    model_path
)

print(
    "Scaler saved:",
    scaler_path
)


# ============================================================
# 25. FINAL BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("FINAL BEST MODEL")
print("=" * 60)

print(
    "\nTuned Random Forest is the final best model."
)

print(
    "Final Accuracy:",
    round(tuned_accuracy, 4)
)

print(
    "Final Precision:",
    round(tuned_precision, 4)
)

print(
    "Final Recall:",
    round(tuned_recall, 4)
)

print(
    "Final F1-Score:",
    round(tuned_f1, 4)
)

print(
    "Final ROC-AUC:",
    round(tuned_roc_auc, 4)
)


print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput files are stored in:")
print(output_dir)

print("\nModel files are stored in:")
print(model_dir)