"""
Model Training Module for Healthcare Readmission Prediction.

Trains a Random Forest classifier and saves the model for deployment.
"""

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_preprocessing import load_data, split_data


def train_model():
    """
    Train Random Forest model on healthcare data.

    Returns:
        dict: Training results including metrics and trained model.
    """
    # Get project root (parent of src)
    BASE_DIR = Path(__file__).parent.parent

    # Load and split data
    df = load_data(BASE_DIR / "data" / "healthcare_readmission_dataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    # Initialize and train Random Forest classifier
    # n_estimators=100, random_state for reproducibility
    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
    print(f"Model saved to: {model_path}")

    return model, results


if __name__ == "__main__":
    print("=" * 50)
    print("Training Random Forest Model...")
    print("=" * 50)

    # Train model and get results
    model, results = train_model()

    # Print evaluation metrics
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"ROC-AUC:  {results['roc_auc']:.4f}")
    print("=" * 50)





