"""
Model Training Module for Healthcare Readmission Prediction.

Trains a Random Forest classifier with hyperparameter tuning and saves
the best model for deployment. Includes MLflow experiment tracking.
"""

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

from src.data_preprocessing import load_data, split_data

# MLflow import (optional - graceful fallback if not installed)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed. Run: pip install mlflow")


def train_model(tune_hyperparams=True):
    """
    Train Random Forest model on healthcare data with optional hyperparameter tuning.

    Args:
        tune_hyperparams: If True, perform GridSearchCV for best parameters.

    Returns:
        tuple: (trained_model, results_dict) with metrics and trained model.
    """
    # Get project root (parent of src)
    BASE_DIR = Path(__file__).parent.parent

    # Load and split data
    df = load_data(BASE_DIR / "data" / "healthcare_readmission_dataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {list(X_train.columns)}")
    print(f"Class balance: {y_train.value_counts().to_dict()}")

    if tune_hyperparams:
        print("\nPerforming hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }

        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)

    # Generate predictions on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    # Save trained model to disk
    model_path = BASE_DIR / "models" / "model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))

    # MLflow Logging
    if MLFLOW_AVAILABLE:
        try:
            # Set up MLflow
            mlflow.set_experiment("Healthcare-Readmission-Prediction")
            tracking_dir = BASE_DIR / "mlruns"
            mlflow.set_tracking_uri(f"file:///{tracking_dir}")

            with mlflow.start_run(run_name="RandomForest_Training"):
                # Log parameters
                if tune_hyperparams and hasattr(model, 'get_params'):
                    params = model.get_params()
                    for param_name, param_value in params.items():
                        mlflow.log_param(param_name, param_value)
                else:
                    mlflow.log_param("n_estimators", 200)
                    mlflow.log_param("max_depth", 15)
                    mlflow.log_param("class_weight", "balanced")

                mlflow.log_param("tune_hyperparams", tune_hyperparams)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))

                # Log metrics
                mlflow.log_metric("accuracy", results['accuracy'])
                mlflow.log_metric("precision", results['precision'])
                mlflow.log_metric("recall", results['recall'])
                mlflow.log_metric("f1_score", results['f1_score'])
                mlflow.log_metric("roc_auc", results['roc_auc'])

                # Log model artifact
                mlflow.sklearn.log_model(model, "model")

                print(f"\n✓ MLflow run logged successfully")
                print(f"  View at: mlflow ui --backend-store-uri file:///{tracking_dir}")
        except Exception as e:
            print(f"\n⚠ MLflow logging failed: {e}")

    return model, results


if __name__ == "__main__":
    print("=" * 60)
    print("Training Random Forest Model (with Hyperparameter Tuning)...")
    print("=" * 60)

    # Train model with hyperparameter tuning
    model, results = train_model(tune_hyperparams=True)

    # Print evaluation metrics
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print("=" * 60)
