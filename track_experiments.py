# ============================================================
# CUSTOMER CHURN MLOPS
# MLFLOW EXPERIMENT TRACKING
# Models:
# 1. Logistic Regression Baseline
# 2. Logistic Regression Tuned
# ============================================================

import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path

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

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "mlflow_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("\n========== MLFLOW CONFIGURATION ==========")
print("Tracking URI:", mlflow.get_tracking_uri())

# ============================================================
# 2. LOAD AND PREPARE DATASET
# ============================================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(DATA_PATH)

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
    raise ValueError("Unexpected target values found.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ============================================================
# 3. DEFINE MODELS TO TRACK
# ============================================================

models_to_track = [
    {
        "name": "Logistic-Regression-Baseline",
        "file": MODEL_DIR / "logistic_regression_pipeline.joblib",
        "classifier_step": "classifier",
        "extra_artifacts": []
    },
    {
        "name": "Logistic-Regression-Tuned",
        "file": MODEL_DIR / "logistic_regression_tuned_pipeline.joblib",
        "classifier_step": "model",
        "extra_artifacts": [
            MODEL_DIR / "cross_validation_results.csv"
        ]
    }
]

# ============================================================
# 4. TRACK EACH MODEL
# ============================================================

for model_config in models_to_track:

    experiment_name = model_config["name"]
    model_path = model_config["file"]
    classifier_step = model_config["classifier_step"]

    print("\n" + "=" * 60)
    print("TRACKING:", experiment_name)
    print("=" * 60)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    # --------------------------------------------------------
    # LOAD SAVED MODEL
    # --------------------------------------------------------

    model = joblib.load(model_path)

    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "precision": float(
            precision_score(y_test, y_pred)
        ),
        "recall": float(
            recall_score(y_test, y_pred)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_prob)
        )
    }

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ]
    )

    # --------------------------------------------------------
    # CREATE / SELECT MLFLOW EXPERIMENT
    # --------------------------------------------------------

    mlflow.set_experiment(experiment_name)

    # --------------------------------------------------------
    # START MLFLOW RUN
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=experiment_name
    ):

        # ----------------------------------------------------
        # GET CLASSIFIER
        # ----------------------------------------------------

        classifier = model.named_steps[
            classifier_step
        ]

        # ----------------------------------------------------
        # LOG MODEL PARAMETERS
        # ----------------------------------------------------

        mlflow.log_params({
            "algorithm": "Logistic Regression",
            "C": classifier.C,
            "class_weight": str(
                classifier.class_weight
            ),
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
            "classification_threshold": 0.5,
            "test_size": 0.20,
            "split_random_state": 42,
            "model_type": (
                "Tuned"
                if "Tuned" in experiment_name
                else "Baseline"
            ),
            "cross_validation": (
                "5-fold StratifiedKFold"
                if "Tuned" in experiment_name
                else "Not applied"
            )
        })

        # ----------------------------------------------------
        # LOG METRICS
        # ----------------------------------------------------

        mlflow.log_metrics(metrics)

        # ----------------------------------------------------
        # SAVE LOCAL EVALUATION ARTIFACTS
        # ----------------------------------------------------

        experiment_output = (
            OUTPUT_DIR / experiment_name
        )

        experiment_output.mkdir(
            exist_ok=True
        )

        # Metrics JSON
        metrics_file = (
            experiment_output / "metrics.json"
        )

        with open(
            metrics_file,
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        # Confusion matrix
        cm_file = (
            experiment_output / "confusion_matrix.csv"
        )

        pd.DataFrame(
            cm,
            index=[
                "Actual_No_Churn",
                "Actual_Churn"
            ],
            columns=[
                "Predicted_No_Churn",
                "Predicted_Churn"
            ]
        ).to_csv(cm_file)

        # Classification report
        report_file = (
            experiment_output /
            "classification_report.txt"
        )

        with open(
            report_file,
            "w"
        ) as file:

            file.write(report)

        # ----------------------------------------------------
        # LOG ARTIFACTS TO MLFLOW
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(metrics_file)
        )

        mlflow.log_artifact(
            str(cm_file)
        )

        mlflow.log_artifact(
            str(report_file)
        )

        # Tuned model CV results
        for artifact in model_config[
            "extra_artifacts"
        ]:

            if artifact.exists():

                mlflow.log_artifact(
                    str(artifact)
                )

        # ----------------------------------------------------
        # LOG COMPLETE SCIKIT-LEARN PIPELINE
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        print("\nLogged parameters:")

        print(
            "Algorithm:",
            "Logistic Regression"
        )

        print(
            "C:",
            classifier.C
        )

        print(
            "Class weight:",
            classifier.class_weight
        )

        print("\nLogged metrics:")

        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.4f}"
            )

        print("\nConfusion matrix:")

        print(cm)

        print("\nMLflow Run ID:")

        print(
            mlflow.active_run()
            .info
            .run_id
        )

    print(
        "\nExperiment completed:",
        experiment_name
    )

# ============================================================
# 5. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("ALL EXPERIMENTS TRACKED SUCCESSFULLY")
print("=" * 60)

print("\nOpen MLflow UI:")
print("http://127.0.0.1:5000")