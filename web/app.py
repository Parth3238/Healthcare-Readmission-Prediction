"""
Flask Web Application for Healthcare Readmission Prediction.

Provides web interface and REST API for patient readmission predictions,
with SHAP-based explanations and session-based prediction history.
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS  # Enable CORS for cross-origin requests
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Get absolute path to web directory
WEB_DIR = Path(__file__).parent.resolve()
BASE_DIR = WEB_DIR.parent.resolve()

# Create Flask app with absolute paths
app = Flask(__name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static")
)

# Secret key for session management (prediction history)
app.secret_key = os.environ.get('SECRET_KEY', 'healthcare-readmission-dev-key-2024')

# Enable CORS for API endpoints (allows cross-origin requests)
CORS(app)

# Feature names expected by the model (for validation)
FEATURES = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient"
]

# Human-readable feature labels for display
FEATURE_LABELS = {
    "age": "Patient Age",
    "time_in_hospital": "Time in Hospital (days)",
    "num_lab_procedures": "Lab Procedures",
    "num_medications": "Medications",
    "number_outpatient": "Outpatient Visits",
    "number_emergency": "Emergency Visits",
    "number_inpatient": "Inpatient Visits"
}

# Feature valid ranges for input validation (min, max)
FEATURE_RANGES = {
    "age": (0, 120),
    "time_in_hospital": (0, 365),
    "num_lab_procedures": (0, 500),
    "num_medications": (0, 200),
    "number_outpatient": (0, 100),
    "number_emergency": (0, 50),
    "number_inpatient": (0, 50)
}

# Load model from parent directory
model_path = BASE_DIR / "models" / "model.pkl"
try:
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

# Try to load SHAP explainer (optional — graceful fallback)
shap_explainer = None
try:
    import shap
    if model is not None:
        # Use TreeExplainer for Random Forest (fast)
        shap_explainer = shap.TreeExplainer(model)
        print("SHAP explainer initialized successfully")
except ImportError:
    print("SHAP not installed — explanations will be disabled. Install with: pip install shap")
except Exception as e:
    print(f"SHAP initialization failed: {e}")


def validate_input(data):
    """
    Validate input data for prediction.

    Args:
        data: Dictionary containing feature values.

    Returns:
        tuple: (is_valid, error_message) - Validation result.
    """
    # Check for missing fields
    for feature in FEATURES:
        if feature not in data:
            return False, f"Missing required field: {feature}"

        # Check if value is numeric
        try:
            value = float(data[feature])
        except (ValueError, TypeError):
            return False, f"Field '{feature}' must be a number"

        # Check value ranges
        min_val, max_val = FEATURE_RANGES[feature]
        if value < min_val or value > max_val:
            return False, f"Field '{feature}' must be between {min_val} and {max_val}"

    return True, None


def get_shap_explanation(input_df):
    """
    Generate SHAP-based explanation for a prediction.

    Args:
        input_df: Single-row DataFrame with patient features.

    Returns:
        list: Top 3 contributing factors with name, value, and importance.
    """
    if shap_explainer is None:
        return None

    try:
        shap_values = shap_explainer.shap_values(input_df)

        # Handle different SHAP output formats:
        # - Newer SHAP: 3D numpy array (samples, features, classes)
        # - Older SHAP: list of 2D arrays [class0_array, class1_array]
        if isinstance(shap_values, list):
            # List format: shap_values[class_index][sample_index]
            values = shap_values[1][0]  # Class 1 (readmitted) for first sample
        elif len(shap_values.shape) == 3:
            # 3D array format: shap_values[sample, feature, class]
            values = shap_values[0, :, 1]  # First sample, all features, class 1
        else:
            # 2D array format: shap_values[sample, feature]
            values = shap_values[0]

        # Build factor list sorted by absolute importance
        factors = []
        for i, feature in enumerate(FEATURES):
            factors.append({
                'name': FEATURE_LABELS.get(feature, feature),
                'value': float(values[i]),
                'importance': min(abs(float(values[i])) * 500, 100)  # Scale for bar width
            })

        # Sort by absolute value (most impactful first)
        factors.sort(key=lambda x: abs(x['value']), reverse=True)

        return factors[:3]  # Top 3
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_to_history(data, result):
    """Add a prediction to the session-based history (max 10 entries)."""
    if 'prediction_history' not in session:
        session['prediction_history'] = []

    entry = {
        'age': data['age'][0],
        'time_in_hospital': data['time_in_hospital'][0],
        'num_lab_procedures': data['num_lab_procedures'][0],
        'num_medications': data['num_medications'][0],
        'number_outpatient': data['number_outpatient'][0],
        'number_emergency': data['number_emergency'][0],
        'number_inpatient': data['number_inpatient'][0],
        'prediction': result['prediction'],
        'prediction_text': result['prediction_text'],
        'confidence': result.get('confidence', 'N/A')
    }

    history = session['prediction_history']
    history.insert(0, entry)  # Most recent first
    session['prediction_history'] = history[:10]  # Keep last 10


@app.route("/")
def home():
    """Render the home page with prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle form-based prediction requests.

    Returns:
        Rendered result.html template with prediction results and SHAP explanations.
    """
    try:
        # Check if model is loaded
        if model is None:
            return render_template("error.html", error="Model not loaded. Please train the model first.")

        # Parse form data
        try:
            data = {
                "age": [float(request.form["age"])],
                "time_in_hospital": [float(request.form["time_in_hospital"])],
                "num_lab_procedures": [float(request.form["num_lab_procedures"])],
                "num_medications": [float(request.form["num_medications"])],
                "number_outpatient": [float(request.form["number_outpatient"])],
                "number_emergency": [float(request.form["number_emergency"])],
                "number_inpatient": [float(request.form["number_inpatient"])]
            }
        except (ValueError, KeyError) as e:
            return render_template("error.html", error=f"Invalid input: {str(e)}")

        # Validate input ranges
        for feature in FEATURES:
            value = data[feature][0]
            min_val, max_val = FEATURE_RANGES[feature]
            if value < min_val or value > max_val:
                return render_template("error.html",
                    error=f"Value for '{feature}' must be between {min_val} and {max_val}")

        # Create DataFrame and make prediction
        df = pd.DataFrame(data)
        prediction = model.predict(df)[0]

        # Get probability scores if available
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(df)[0]

        result = {
            "prediction": int(prediction),
            "prediction_text": "Readmitted" if prediction == 1 else "Not Readmitted",
            "confidence": round(float(max(prediction_proba)) * 100, 2) if prediction_proba is not None else None
        }

        # Generate SHAP explanation
        shap_factors = get_shap_explanation(df)

        # Save to history
        add_to_history(data, result)

        return render_template("result.html", result=result, form_data=data, shap_factors=shap_factors)

    except Exception as e:
        return render_template("error.html", error=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic prediction access.

    Expected JSON input:
    {
        "age": 65,
        "time_in_hospital": 4,
        ...
    }

    Returns:
        JSON response with prediction, confidence score, and SHAP explanations.
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Please train the model first."}), 503

        # Parse JSON input
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON. Please provide valid JSON body."}), 400

        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        # Create DataFrame with validated data
        df = pd.DataFrame({k: [float(v)] for k, v in data.items()})

        # Make prediction
        prediction = model.predict(df)[0]

        # Get probability scores
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(df)[0]

        # Build response
        response = {
            "prediction": int(prediction),
            "prediction_text": "Readmitted" if prediction == 1 else "Not Readmitted",
            "confidence": round(float(max(prediction_proba)), 4) if prediction_proba is not None else None
        }

        # Add SHAP explanations if available
        shap_factors = get_shap_explanation(df)
        if shap_factors:
            response["contributing_factors"] = [
                {"feature": f["name"], "impact": round(f["value"], 4)}
                for f in shap_factors
            ]

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/history")
def history():
    """Display the prediction history from the current session."""
    prediction_history = session.get('prediction_history', [])
    return render_template("history.html", history=prediction_history)


if __name__ == "__main__":
    print(f"Model loaded from: {model_path}")
    print(f"Templates folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    print("Server starting at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
