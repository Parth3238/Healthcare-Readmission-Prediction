"""
MLflow Experiment Tracking for Healthcare Readmission Prediction.

Logs model training runs with metrics (Accuracy, F1, ROC-AUC),
parameters, and artifacts for reproducibility and comparison.
"""

import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import load_data, split_data

BASE_DIR = Path(__file__).parent.parent


def train_and_log(model, model_name, X_train, X_test, y_train, y_test, params=None):
    """
    Train a model and log everything to MLflow.

    Args:
        model: sklearn model instance.
        model_name: Name for the run.
        X_train, X_test, y_train, y_test: Train/test splits.
        params: Dictionary of model parameters to log.

    Returns:
        dict: Computed metrics.
    """
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        if params:
            mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted'),
            "recall_weighted": recall_score(y_test, y_pred, average='weighted'),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
            "f1_binary": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        # Print summary
        print(f"\n{'='*50}")
        print(f"Run: {model_name}")
        print(f"{'='*50}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return metrics


def run_experiment():
    """Run a full MLflow experiment comparing multiple models."""
    # Set experiment name
    mlflow.set_experiment("Healthcare-Readmission-Prediction")

    # Set tracking URI to local directory
    tracking_dir = BASE_DIR / "mlruns"
    mlflow.set_tracking_uri(f"file:///{tracking_dir}")

    print("=" * 60)
    print("MLFLOW EXPERIMENT TRACKING")
    print(f"Tracking URI: {tracking_dir}")
    print("=" * 60)

    # Load data
    df = load_data(BASE_DIR / "data" / "healthcare_readmission_dataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Dataset: {len(df)} samples")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # --- Run 1: Random Forest (Default) ---
    train_and_log(
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        model_name="RandomForest_Default",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params={"n_estimators": 100, "max_depth": "None", "random_state": 42}
    )

    # --- Run 2: Random Forest (Tuned) ---
    train_and_log(
        model=RandomForestClassifier(
            n_estimators=300, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced', random_state=42
        ),
        model_name="RandomForest_Tuned",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params={
            "n_estimators": 300, "max_depth": 20,
            "min_samples_split": 5, "min_samples_leaf": 2,
            "class_weight": "balanced"
        }
    )

    # --- Run 3: Gradient Boosting ---
    train_and_log(
        model=GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.1, random_state=42
        ),
        model_name="GradientBoosting",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params={
            "n_estimators": 200, "max_depth": 5,
            "learning_rate": 0.1
        }
    )

    # --- Run 4: Logistic Regression (Baseline) ---
    train_and_log(
        model=LogisticRegression(max_iter=1000, random_state=42),
        model_name="LogisticRegression_Baseline",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params={"max_iter": 1000, "solver": "lbfgs"}
    )

    print("\n" + "=" * 60)
    print("ALL RUNS COMPLETE")
    print(f"View results: mlflow ui --backend-store-uri file:///{tracking_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
