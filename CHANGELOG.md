# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-03-14

### Added
- Initial release of Healthcare Readmission Prediction System
- Random Forest Classifier for readmission prediction with hyperparameter tuning
- Flask web application with premium hospital-themed UI
- REST API for programmatic access with input validation
- Data preprocessing pipeline with train/test splitting
- Model evaluation with comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- **SHAP values for model explainability** — top 3 contributing factors per prediction
- **MLflow integration for experiment tracking** — logs metrics, parameters, and artifacts
- Feature importance visualization
- ROC curve and confusion matrix plots
- Cross-validation for model stability
- Jupyter notebook for EDA and model comparison
- Model comparison across 8 ML algorithms
- Prediction history with session-based storage
- Comprehensive test suite (unit, integration, edge cases)
- Docker support with Dockerfile and docker-compose
- 10,000-sample realistic synthetic dataset
- Comprehensive documentation with README
- MIT License

### Features
- Web interface with gradient design and smooth animations
- Real-time prediction with confidence scores
- SHAP-based explanation of predictions (top 3 factors)
- Session-based prediction history (last 10 predictions)
- API endpoint with JSON responses
- Patient data visualization
- Error handling and validation
- Mobile responsive design

### Technical
- Python 3.9+ support
- scikit-learn for ML with GridSearchCV hyperparameter tuning
- Flask for web framework with CORS support
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization
- Joblib for model serialization
- SHAP for model interpretability
- MLflow for experiment tracking

## [Unreleased]

### Planned
- XGBoost and Neural Network models
- Automated retraining pipeline
- Mobile app for healthcare providers
- FHIR integration for EHR systems
- Real-time monitoring dashboard
- Kubernetes deployment support
- CI/CD pipeline with GitHub Actions
