
# ============================================================
# CUSTOMER CHURN PREDICTION
# LOGISTIC REGRESSION: CROSS-VALIDATION AND HYPERPARAMETER TUNING
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "Telco-Customer-Churn.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n========== DATA CLEANING ==========")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected values found in Churn column.")

print("Missing values after cleaning:", df.isnull().sum().sum())


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    column for column in X.columns
    if column not in numerical_features
]


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

print("\n========== TRAIN-TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 7. LOGISTIC REGRESSION PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=42
            )
        )
    ]
)


# ============================================================
# 8. DEFINE HYPERPARAMETER SEARCH
# ============================================================

parameter_grid = {
    "model__C": [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ],

    "model__class_weight": [
        None,
        "balanced"
    ]
}


# ============================================================
# 9. STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================

cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 10. GRID SEARCH
# ============================================================

print("\n========== STARTING HYPERPARAMETER TUNING ==========")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=cross_validation,
    n_jobs=1,
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)


# ============================================================
# 11. BEST MODEL INFORMATION
# ============================================================

best_model = grid_search.best_estimator_

print("\n========== TUNING RESULTS ==========")

print("Best cross-validation F1-score:")
print(round(grid_search.best_score_, 4))

print("\nBest hyperparameters:")

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")


# ============================================================
# 12. EVALUATE BEST MODEL ON TEST DATA
# ============================================================

print("\n========== TEST SET EVALUATION ==========")

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
)

print(report)


# ============================================================
# 15. SAVE BEST MODEL
# ============================================================

model_path = MODEL_DIR / "logistic_regression_tuned_pipeline.joblib"

joblib.dump(best_model, model_path)

print("\nTuned model saved to:")
print(model_path)


# ============================================================
# 16. SAVE EVALUATION METRICS
# ============================================================

metrics = {
    "best_cv_f1_score": float(grid_search.best_score_),
    "best_parameters": grid_search.best_params_,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist()
}

metrics_path = MODEL_DIR / "tuned_evaluation_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nEvaluation metrics saved to:")
print(metrics_path)


# ============================================================
# 17. SAVE CROSS-VALIDATION RESULTS
# ============================================================

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results.to_csv(
    MODEL_DIR / "cross_validation_results.csv",
    index=False
)

print("\nCross-validation results saved.")

print("\n========== TUNING COMPLETED ==========")
