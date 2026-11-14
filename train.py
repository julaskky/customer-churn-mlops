
# ============================================================
# CUSTOMER CHURN PREDICTION
# LOGISTIC REGRESSION MODEL TRAINING
# ============================================================

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "Telco-Customer-Churn.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\n========== LOADING DATASET ==========\n")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ------------------------------------------------------------
# 3. DATA CLEANING
# ------------------------------------------------------------

print("\n========== DATA CLEANING ==========\n")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Replace missing TotalCharges with zero
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove customer identifier
df = df.drop(columns=["customerID"])

# Encode target variable
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if df["Churn"].isnull().any():
    raise ValueError("Unexpected values found in Churn column.")

print("Missing values after cleaning:", df.isnull().sum().sum())

print("Cleaned dataset shape:", df.shape)


# ------------------------------------------------------------
# 4. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

print("\n========== FEATURE SELECTION ==========\n")

X = df.drop(columns=["Churn"])

y = df["Churn"]

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    column
    for column in X.columns
    if column not in numerical_features
]

print("Number of input features:", X.shape[1])

print("Numerical features:", numerical_features)

print("Categorical features:", categorical_features)

print("\nTarget distribution:")

print(y.value_counts())


# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("\n========== TRAIN-TEST SPLIT ==========\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training features:", X_train.shape)

print("Testing features:", X_test.shape)

print("\nTraining target distribution:")

print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")

print(y_test.value_counts(normalize=True))


# ------------------------------------------------------------
# 6. BUILD PREPROCESSING PIPELINE
# ------------------------------------------------------------

print("\n========== BUILDING PREPROCESSOR ==========\n")

numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_transformer,
            numerical_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 7. BUILD LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

print("\n========== BUILDING LOGISTIC REGRESSION ==========\n")

logistic_regression = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    random_state=RANDOM_STATE
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", logistic_regression)
    ]
)


# ------------------------------------------------------------
# 8. TRAIN MODEL
# ------------------------------------------------------------

print("\n========== MODEL TRAINING ==========\n")

model.fit(X_train, y_train)

print("Logistic Regression training completed.")


# ------------------------------------------------------------
# 9. MAKE PREDICTIONS
# ------------------------------------------------------------

print("\n========== MODEL PREDICTIONS ==========\n")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 10. MODEL EVALUATION
# ------------------------------------------------------------

print("\n========== MODEL EVALUATION ==========\n")

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

f1 = f1_score(y_test, y_pred, zero_division=0)

roc_auc = roc_auc_score(y_test, y_probability)


print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1-Score:  {f1:.4f}")

print(f"ROC-AUC:   {roc_auc:.4f}")


# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

print("\n========== CONFUSION MATRIX ==========\n")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ------------------------------------------------------------
# 12. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 13. SAVE MODEL
# ------------------------------------------------------------

print("\n========== SAVING MODEL ==========\n")

joblib.dump(model, MODEL_PATH)

print("Model saved successfully.")

print("Model location:", MODEL_PATH)


# ------------------------------------------------------------
# 14. SAVE EVALUATION METRICS
# ------------------------------------------------------------

metrics = {
    "algorithm": "Logistic Regression",
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE
}

metrics_path = MODEL_DIR / "evaluation_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("Evaluation metrics saved:", metrics_path)


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n========== TRAINING COMPLETED ==========\n")

print("Algorithm: Logistic Regression")

print("Training records:", len(X_train))

print("Testing records:", len(X_test))

print("Model artefacts:", MODEL_DIR)
