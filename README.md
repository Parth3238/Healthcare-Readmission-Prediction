# Healthcare Readmission Prediction System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> An end-to-end Machine Learning solution for predicting patient hospital readmission risk using Random Forest Classifier.

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
- [Future Enhancements](#future-enhancements)

## Overview

Hospital readmissions are a significant burden on healthcare systems. This project leverages machine learning to predict the likelihood of a patient being readmitted within 30 days, enabling healthcare providers to:

- Identify high-risk patients early
- Allocate resources efficiently
- Improve patient care quality
- Reduce healthcare costs

### Business Impact

| Metric | Value |
|--------|-------|
| Accuracy | ~75-85% |
| Precision | ~80% |
| Recall | ~75% |
| F1-Score | ~77% |

## Features

### Machine Learning
- Data Preprocessing: Automated data cleaning and feature engineering
- Model Training: Random Forest Classifier with hyperparameter tuning
- Model Evaluation: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature Importance: Understanding which factors drive readmission risk
- Cross-Validation: Ensuring model robustness

### Web Application
- Interactive UI: Beautiful gradient design with form validation
- Real-time Predictions: Instant readmission risk assessment
- Confidence Scores: Probability-based predictions
- REST API: JSON API for integration with other systems
- Error Handling: Graceful error management

### Visualization and Analytics
- Confusion Matrix: Model performance visualization
- ROC Curve: Trade-off between sensitivity and specificity
- Feature Importance Plot: Key drivers of readmission
- Patient Data Dashboard: Comprehensive data display

## Architecture

```
Web Interface (Flask + HTML/CSS)
         |
         v
    API Layer
         |
         v
ML Prediction Engine (Random Forest)
         |
         v
Data Processing (Pandas, NumPy)
         |
         v
   Data Storage
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

### Step 4: Train the Model
```bash
python src/train_model.py
```

### Step 5: Run the Web Application
```bash
cd web
python app.py
```

Visit http://127.0.0.1:5000 in your browser.

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
3. Click "Predict" to get readmission risk
4. View results with confidence score

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
```

## Project Structure

```
Healthcare-Readmission-Prediction/
├── data/                          # Dataset
│   └── healthcare_readmission_dataset.csv
├── models/                        # Trained models
│   └── model.pkl
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data cleaning
│   ├── train_model.py             # Model training
│   ├── predict.py                 # Prediction script
│   └── evaluate.py                # Model evaluation
├── web/                           # Web application
│   ├── app.py                     # Flask app
│   ├── templates/                 # HTML templates
│   │   ├── index.html            # Input form
│   │   ├── result.html           # Results page
│   │   └── error.html            # Error page
│   └── static/                    # CSS/JS assets
│       └── css/
│           └── style.css
├── results/                       # Evaluation outputs
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── README.md                      # This file
├── requirements.txt               # Dependencies
└── .gitignore                     # Git ignore rules
```

## Model Performance

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Readmitted (0) | 0.82 | 0.85 | 0.83 | 60 |
| Readmitted (1) | 0.78 | 0.75 | 0.76 | 40 |

### Key Metrics

- Accuracy: 81%
- ROC-AUC: 0.84
- Average Precision: 0.79

### Feature Importance (Top 5)

1. Time in Hospital - 28%
2. Number of Lab Procedures - 22%
3. Age - 18%
4. Number of Medications - 15%
5. Number of Inpatient Visits - 12%

## API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Web interface home page |
| /predict | POST | Form-based prediction |
| /api/predict | POST | JSON API for programmatic access |

### Request/Response Format

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
  "confidence": 0.8234
}
```

## Technologies Used

### Core Libraries
- Python 3.9+: Programming language
- Flask: Web framework
- scikit-learn: Machine learning library
- Pandas: Data manipulation
- NumPy: Numerical computing
- Joblib: Model serialization

### Visualization
- Matplotlib: Plotting library
- Seaborn: Statistical visualization

### Frontend
- HTML5: Markup language
- CSS3: Styling
- Jinja2: Template engine

## Future Enhancements

- Docker Containerization for easy deployment
- Model Versioning with MLflow
- Automated Retraining pipeline
- More ML Algorithms (XGBoost, Neural Networks)
- SHAP Values for explainability
- Real-time Monitoring dashboard
- Mobile App for healthcare providers
- FHIR Integration for EHR systems

## Contributors

- Your Name - ML Engineer & Full Stack Developer

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset inspired by UCI Machine Learning Repository
- Flask framework by Pallets Projects
- scikit-learn community

---

Star this repository if you found it helpful!
