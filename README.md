# Healthcare Readmission Prediction System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> An end-to-end Machine Learning solution for predicting patient hospital readmission risk using Random Forest Classifier, with SHAP-based explainability and MLflow experiment tracking.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Technologies Used](#technologies-used)
- [Testing](#testing)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

## Overview

Hospital readmissions are a significant burden on healthcare systems, costing the U.S. healthcare system over $26 billion annually. This project leverages machine learning to predict the likelihood of a patient being readmitted within 30 days, enabling healthcare providers to:

- Identify high-risk patients early
- Allocate resources efficiently
- Improve patient care quality
- Reduce healthcare costs

### Business Impact

This system provides actionable risk assessment with explainable predictions, helping clinicians understand *why* a patient is flagged as high-risk through SHAP-based feature importance analysis.

## Features

### Machine Learning
- **Data Pipeline**: Automated data loading, preprocessing, and train/test splitting
- **Model Training**: Random Forest Classifier with hyperparameter tuning (GridSearchCV)
- **Model Comparison**: 8 algorithm comparison (RF, GBM, Logistic Regression, SVM, KNN, Decision Tree, AdaBoost, Naive Bayes)
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Explainability**: SHAP values for model interpretability — top 3 contributing factors per prediction
- **Experiment Tracking**: MLflow integration for logging metrics, parameters, and model artifacts

### Web Application
- **Premium UI**: Hospital-themed design with Inter font, card layouts, and gradient backgrounds
- **Real-time Predictions**: Instant readmission risk assessment with confidence scores
- **SHAP Explanations**: Visual display of top factors contributing to each prediction
- **Prediction History**: Session-based storage of last 10 predictions
- **REST API**: JSON API endpoint for programmatic access
- **API Documentation**: Built-in expandable API docs section
- **Input Validation**: Range-checked form inputs matching backend constraints
- **Mobile Responsive**: Fully responsive design for all screen sizes

### Dataset
- **10,000 samples** generated with clinically realistic distributions
- **7 clinical features**: Age, Time in Hospital, Lab Procedures, Medications, Outpatient/Emergency/Inpatient Visits
- **29.1% readmission rate** — mirrors real-world clinical readmission patterns
- **Logistic risk model**: Target variable generated from evidence-based clinical risk factors

## Architecture

```
┌─────────────────────────────────┐
│   Web Interface (Flask + HTML)  │
│   Premium UI with SHAP display  │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│       API Layer (REST JSON)     │
│   /predict  |  /api/predict     │
│   /history                      │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│  ML Prediction Engine           │
│  Random Forest + SHAP Explainer │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│  Data Processing (Pandas/NumPy) │
│  Preprocessing + Validation     │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│  MLflow Experiment Tracking     │
│  Metrics, Parameters, Artifacts │
└─────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Healthcare-Readmission-Prediction.git
cd Healthcare-Readmission-Prediction
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Generate Dataset (Optional — pre-generated dataset included)
```bash
python src/generate_dataset.py
```

### Step 5: Train the Model
```bash
python src/train_model.py
```

### Step 6: Run the Web Application
```bash
cd web
python app.py
```

Visit http://127.0.0.1:5000 in your browser.

### Step 7 (Optional): Run MLflow Experiment Tracking
```bash
python src/mlflow_tracking.py
mlflow ui
```

## Usage

### Web Interface

1. Access the application at http://127.0.0.1:5000
2. Enter patient data:
   - Age
   - Time in Hospital (days)
   - Number of Lab Procedures
   - Number of Medications
   - Outpatient Visits
   - Emergency Visits
   - Inpatient Visits
3. Click **"Predict Readmission Risk"**
4. View results with:
   - Color-coded risk badge (red = high risk, green = low risk)
   - Confidence score
   - Top 3 SHAP contributing factors
   - Interactive probability chart
5. View past predictions on the **History** page

### API Usage

```python
import requests

url = "http://127.0.0.1:5000/api/predict"
data = {
    "age": 65,
    "time_in_hospital": 4,
    "num_lab_procedures": 40,
    "num_medications": 10,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 0
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['prediction_text']}")
print(f"Confidence: {result['confidence']}")
print(f"Top Factors: {result.get('contributing_factors', [])}")
```

### API Example (cURL)

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "time_in_hospital": 4, "num_lab_procedures": 40, "num_medications": 10, "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 0}'
```

## Project Structure

```
Healthcare-Readmission-Prediction/
├── data/                              # Dataset
│   └── healthcare_readmission_dataset.csv  (10,000 rows)
├── models/                            # Trained models
│   └── model.pkl                      # Primary trained model
├── src/                               # Source code
│   ├── __init__.py                    # Package init
│   ├── data_preprocessing.py          # Data loading and splitting
│   ├── train_model.py                 # Model training with GridSearchCV
│   ├── predict.py                     # Reusable prediction module
│   ├── evaluate.py                    # Model evaluation with metrics & plots
│   ├── model_comparison.py            # 8-model comparison
│   ├── generate_dataset.py            # Realistic dataset generator
│   └── mlflow_tracking.py            # MLflow experiment tracking
├── web/                               # Web application
│   ├── app.py                         # Flask app (REST API + SHAP)
│   ├── templates/                     # HTML templates
│   │   ├── index.html                 # Input form with API docs
│   │   ├── result.html                # Results + SHAP factors
│   │   ├── error.html                 # Error page
│   │   └── history.html               # Prediction history
│   └── static/                        # CSS assets
│       └── css/
│           └── style.css              # Premium hospital-themed CSS
├── tests/                             # Test suite
│   ├── test_models.py                 # Data and model unit tests
│   ├── test_prediction.py             # Prediction function tests
│   └── test_flask_routes.py           # Flask route & edge case tests
├── results/                           # Evaluation outputs
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── model_comparison_results.csv
├── notebooks/
│   └── EDA_and_Model_Comparison.ipynb # Exploratory data analysis
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── setup.py                           # Package setup
├── Dockerfile                         # Docker container
├── docker-compose.yml                 # Docker compose
└── Makefile                           # Build automation
```

## Model Performance

### Honest Assessment

This project uses a **10,000-sample dataset** with realistic clinical distributions. The readmission prediction task is inherently challenging — even state-of-the-art models in published research typically achieve 65–75% accuracy on this problem due to the complex, multi-factorial nature of hospital readmissions.

### Training Results (Random Forest — 300 trees)

| Metric | Score |
|--------|-------|
| **Accuracy** | ~62-72%* |
| **Precision (weighted)** | ~60-68% |
| **Recall (weighted)** | ~62-72% |
| **F1-Score (weighted)** | ~55-66% |
| **ROC-AUC** | ~0.60-0.65 |

**Note**: Results vary between 62-72% depending on the random train/test split. The model comparison results
show ~63% accuracy while the tuned model achieves ~72% on favorable splits. This variance is normal for
healthcare prediction tasks with limited feature sets.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Readmitted (0) | 0.74 | 0.93 | 0.82 | ~1417 |
| Readmitted (1) | 0.53 | 0.19 | 0.28 | ~583 |

### Why These Numbers Are Realistic

- **Hospital readmission prediction is a well-known hard problem** — published research on the UCI Diabetes 130-US Hospitals dataset reports similar accuracy ranges (60–75%)
- **Class imbalance**: Only ~29% of patients are readmitted, making the minority class harder to predict
- **Limited features**: We use 7 demographic/clinical features; real EHR systems have hundreds
- **This is a prototype**: Production systems would incorporate diagnosis codes (ICD), lab results, medications, social determinants of health, and temporal patterns

### Feature Importance (Top 5)

1. **Number of Inpatient Visits** — Prior inpatient stays are the strongest signal
2. **Number of Emergency Visits** — ER utilization indicates instability
3. **Time in Hospital** — Longer stays correlate with readmission risk
4. **Age** — Older patients face higher readmission risk
5. **Number of Medications** — Polypharmacy is a known risk factor

## API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface home page |
| `/predict` | POST | Form-based prediction (returns HTML) |
| `/api/predict` | POST | JSON API for programmatic access |
| `/history` | GET | View last 10 predictions |
| `/docs` | GET | Full API documentation page |

### API Request/Response Format

**Request (POST /api/predict):**
```json
{
  "age": 65,
  "time_in_hospital": 4,
  "num_lab_procedures": 40,
  "num_medications": 10,
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 0
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_text": "Readmitted",
  "confidence": 0.7234,
  "contributing_factors": [
    {"feature": "Emergency Visits", "impact": 0.0821},
    {"feature": "Patient Age", "impact": 0.0534},
    {"feature": "Time in Hospital (days)", "impact": 0.0312}
  ]
}
```

### Error Response Format
```json
{
  "error": "Missing required field: age"
}
```

### Input Validation Rules

| Field | Type | Valid Range |
|-------|------|-------------|
| age | integer | 0–120 |
| time_in_hospital | integer | 0–365 |
| num_lab_procedures | integer | 0–500 |
| num_medications | integer | 0–200 |
| number_outpatient | integer | 0–100 |
| number_emergency | integer | 0–50 |
| number_inpatient | integer | 0–50 |

## Technologies Used

### Core Libraries
- **Python 3.9+**: Programming language
- **Flask 3.0+**: Web framework
- **flask-cors**: Cross-origin resource sharing
- **scikit-learn 1.3+**: Machine learning library
- **Pandas 2.1+**: Data manipulation
- **NumPy 1.26+**: Numerical computing
- **Joblib 1.3+**: Model serialization

### Model Explainability & Tracking
- **SHAP 0.43+**: SHapley Additive exPlanations for model interpretability
- **MLflow 2.9+**: Experiment tracking and model registry

### Visualization
- **Matplotlib 3.8+**: Plotting library
- **Seaborn 0.13+**: Statistical visualization
- **Chart.js 4.4**: Interactive web charts

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Premium hospital-themed design with Inter font
- **Jinja2**: Template engine

### Testing
- **pytest**: Unit testing framework
- **Flask test_client()**: Integration testing for routes

## Testing

Run all tests:
```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=web -v

# Run specific test file
pytest tests/test_flask_routes.py -v
```

### Test Coverage
- **Unit Tests**: Data preprocessing, model training, prediction functions
- **Integration Tests**: Flask routes (`/`, `/predict`, `/api/predict`, `/history`)
- **Edge Case Tests**: All-zeros input, maximum values, negative values, string values, boundary conditions
- **Session Tests**: Prediction history storage and retrieval

## Future Enhancements

The following improvements are planned for future versions:

### Authentication & Security
- JWT-based authentication for API endpoints
- Role-based access control (RBAC) for healthcare staff
- HTTPS/TLS encryption for data in transit

### Logging & Monitoring
- Structured logging with Logstash/ELK stack
- Request/response logging for audit trails
- Performance monitoring with Grafana

### Scalability
- Kubernetes deployment for auto-scaling
- Redis caching for model predictions
- Load balancing across multiple Flask instances
- Model serving with TensorFlow Serving or Triton

### Machine Learning
- XGBoost and LightGBM integration
- Automated hyperparameter tuning with Optuna
- A/B testing framework for model comparison
- Real-time model retraining pipeline

## Contributors

| Name | Role | Contributions |
|------|------|--------------|
| **Parth Agrawal** | ML Engineer & Full Stack Developer | Data pipeline, model training, Flask backend, API development, testing |
| **Yagh Chaudhary** | Frontend Developer & ML Researcher | UI/UX design, CSS theming, SHAP integration, documentation, dataset research |

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset generation inspired by the [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296) dataset
- SHAP library by [Scott Lundberg](https://github.com/slundberg/shap)
- Flask framework by Pallets Projects
- scikit-learn community

---

⭐ Star this repository if you found it helpful!
