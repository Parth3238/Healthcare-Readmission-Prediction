"""
Flask Web Application for Healthcare Readmission Prediction.

Provides web interface and REST API for patient readmission predictions.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Enable CORS for cross-origin requests
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Get absolute path to web directory
WEB_DIR = Path(__file__).parent.resolve()
BASE_DIR = WEB_DIR.parent.resolve()

# Create Flask app with absolute paths
app = Flask(__name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static")
)

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


@app.route("/")
def home():
    """Render the home page with prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle form-based prediction requests.

    Returns:
        Rendered result.html template with prediction results.
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

        return render_template("result.html", result=result, form_data=data)

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
        JSON response with prediction and confidence score.
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

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ---
# Authentication placeholder (commented out - implement as needed)
# ---
# from functools import wraps
# def require_auth(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         # TODO: Implement JWT/session-based authentication
#         # Example: check Authorization header, validate token
#         auth_header = request.headers.get('Authorization')
#         if not auth_header or not auth_header.startswith('Bearer '):
#             return jsonify({"error": "Unauthorized"}), 401
#         return f(*args, **kwargs)
#     return decorated


# ---
# Rate limiting placeholder (commented out - implement as needed)
# ---
# TODO: Add rate limiting using Flask-Limiter
# Example: @limiter.limit("100 per minute")
# This prevents abuse by limiting request frequency


if __name__ == "__main__":
    print(f"Model loaded from: {model_path}")
    print(f"Templates folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    print("Server starting at http://127.0.0.1:5000")
    print(" * Running on http://127.0.0.1:5000")
    print(" * Running on http://192.168.0.111:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)





    

