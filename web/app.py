from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from pathlib import Path

# Get absolute path to web directory
WEB_DIR = Path(__file__).parent.resolve()
BASE_DIR = WEB_DIR.parent.resolve()

# Create Flask app with absolute paths
app = Flask(__name__, 
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static")
)

# Load model from parent directory
model_path = BASE_DIR / "models" / "model.pkl"
model = joblib.load(model_path)

# Feature names
FEATURES = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        data = {
            "age": [float(request.form["age"])],
            "time_in_hospital": [float(request.form["time_in_hospital"])],
            "num_lab_procedures": [float(request.form["num_lab_procedures"])],
            "num_medications": [float(request.form["num_medications"])],
            "number_outpatient": [float(request.form["number_outpatient"])],
            "number_emergency": [float(request.form["number_emergency"])],
            "number_inpatient": [float(request.form["number_inpatient"])]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Make prediction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0] if hasattr(model, 'predict_proba') else None
        
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
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        
        # Validate required fields
        for feature in FEATURES:
            if feature not in data:
                return jsonify({"error": f"Missing field: {feature}"}), 400
        
        # Create DataFrame
        df = pd.DataFrame({k: [v] for k, v in data.items()})
        
        # Make prediction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0] if hasattr(model, 'predict_proba') else None
        
        return jsonify({
            "prediction": int(prediction),
            "prediction_text": "Readmitted" if prediction == 1 else "Not Readmitted",
            "confidence": round(float(max(prediction_proba)), 4) if prediction_proba is not None else None
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"Model loaded from: {model_path}")
    print(f"Templates folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    print("Server starting at http://127.0.0.1:5000")
    print(" * Running on http://127.0.0.1:5000")
    print(" * Running on http://192.168.0.111:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)


