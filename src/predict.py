"""
Prediction Module for Healthcare Readmission Prediction.

Provides reusable functions for making patient readmission predictions
using the trained Random Forest model.
"""

import joblib
import pandas as pd
from pathlib import Path

# Get the project root directory (parent of src)
BASE_DIR = Path(__file__).parent.parent

# Feature names expected by the model
FEATURES = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient"
]


def load_model(model_path=None):
    """
    Load the trained model from disk.

    Args:
        model_path: Optional path to model file. Defaults to models/model.pkl.

    Returns:
        Trained sklearn model object.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if model_path is None:
        model_path = BASE_DIR / "models" / "model.pkl"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    return joblib.load(model_path)


def predict_patient(data, model=None):
    """
    Predict readmission risk for a single patient.

    Args:
        data: Dictionary with patient features (age, time_in_hospital, etc.)
        model: Pre-loaded model object. If None, loads from default path.

    Returns:
        dict: {
            'prediction': 0 or 1,
            'prediction_text': 'Readmitted' or 'Not Readmitted',
            'confidence': float (0-100),
            'probabilities': {'not_readmitted': float, 'readmitted': float}
        }

    Raises:
        ValueError: If required features are missing from input data.
    """
    # Validate input
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Load model if not provided
    if model is None:
        model = load_model()

    # Create DataFrame for prediction
    df = pd.DataFrame({k: [float(data[k])] for k in FEATURES})

    # Make prediction
    prediction = model.predict(df)[0]

    # Get probability scores
    result = {
        'prediction': int(prediction),
        'prediction_text': 'Readmitted' if prediction == 1 else 'Not Readmitted',
    }

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df)[0]
        result['confidence'] = round(float(max(proba)) * 100, 2)
        result['probabilities'] = {
            'not_readmitted': round(float(proba[0]), 4),
            'readmitted': round(float(proba[1]), 4)
        }

    return result


def predict_batch(data_list, model=None):
    """
    Predict readmission risk for multiple patients.

    Args:
        data_list: List of dictionaries, each with patient features.
        model: Pre-loaded model object. If None, loads from default path.

    Returns:
        list: List of prediction result dictionaries.
    """
    if model is None:
        model = load_model()

    return [predict_patient(data, model=model) for data in data_list]


if __name__ == "__main__":
    # Example usage
    sample_patient = {
        "age": 65,
        "time_in_hospital": 4,
        "num_lab_procedures": 40,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 0
    }

    print("=" * 50)
    print("Healthcare Readmission Prediction")
    print("=" * 50)

    result = predict_patient(sample_patient)

    print(f"\nPrediction: {result['prediction_text']}")
    print(f"Confidence: {result.get('confidence', 'N/A')}%")

    if 'probabilities' in result:
        print(f"  Not Readmitted: {result['probabilities']['not_readmitted']:.2%}")
        print(f"  Readmitted:     {result['probabilities']['readmitted']:.2%}")
