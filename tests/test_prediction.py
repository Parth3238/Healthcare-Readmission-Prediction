"""
Basic Tests for Healthcare Readmission Prediction System.
Tests prediction functionality and API endpoints.
"""

import unittest
import sys
from pathlib import Path

# Add web directory to path for app imports
sys.path.insert(0, str(Path(__file__).parent.parent / "web"))

import joblib
import pandas as pd
import numpy as np


class TestPredictionFunction(unittest.TestCase):
    """Test cases for the prediction function."""

    @classmethod
    def setUpClass(cls):
        """Load model for testing."""
        models_dir = Path(__file__).parent.parent / "models"
        model_path = models_dir / "model.pkl"

        if model_path.exists():
            cls.model = joblib.load(model_path)
        else:
            cls.model = None

    def test_model_can_predict(self):
        """Test that model can make predictions on valid input."""
        if self.model is None:
            self.skipTest("Model not found - please train model first")

        # Sample patient data
        sample_data = pd.DataFrame({
            "age": [65],
            "time_in_hospital": [4],
            "num_lab_procedures": [40],
            "num_medications": [10],
            "number_outpatient": [0],
            "number_emergency": [1],
            "number_inpatient": [0]
        })

        prediction = self.model.predict(sample_data)
        self.assertIn(prediction[0], [0, 1])

    def test_model_predict_proba(self):
        """Test that model can output probability scores."""
        if self.model is None:
            self.skipTest("Model not found - please train model first")

        sample_data = pd.DataFrame({
            "age": [65],
            "time_in_hospital": [4],
            "num_lab_procedures": [40],
            "num_medications": [10],
            "number_outpatient": [0],
            "number_emergency": [1],
            "number_inpatient": [0]
        })

        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(sample_data)
            self.assertEqual(proba.shape, (1, 2))
            self.assertAlmostEqual(sum(proba[0]), 1.0, places=5)
        else:
            self.skipTest("Model does not support predict_proba")

    def test_prediction_on_batch(self):
        """Test predictions on multiple samples."""
        if self.model is None:
            self.skipTest("Model not found - please train model first")

        batch_data = pd.DataFrame({
            "age": [65, 70, 55, 80, 45],
            "time_in_hospital": [4, 8, 2, 12, 3],
            "num_lab_procedures": [40, 60, 20, 80, 30],
            "num_medications": [10, 15, 5, 20, 8],
            "number_outpatient": [0, 2, 1, 0, 3],
            "number_emergency": [1, 0, 2, 1, 0],
            "number_inpatient": [0, 1, 0, 2, 1]
        })

        predictions = self.model.predict(batch_data)
        self.assertEqual(len(predictions), 5)
        for pred in predictions:
            self.assertIn(pred, [0, 1])


class TestInputValidation(unittest.TestCase):
    """Test input validation for prediction."""

    def test_feature_names_required(self):
        """Test that all required features are present."""
        required_features = [
            "age", "time_in_hospital", "num_lab_procedures",
            "num_medications", "number_outpatient",
            "number_emergency", "number_inpatient"
        ]

        sample_data = {f: [1] for f in required_features}
        df = pd.DataFrame(sample_data)

        for feature in required_features:
            self.assertIn(feature, df.columns)

    def test_dataframe_shape(self):
        """Test that DataFrame has correct shape for single prediction."""
        sample_data = {
            "age": [65],
            "time_in_hospital": [4],
            "num_lab_procedures": [40],
            "num_medications": [10],
            "number_outpatient": [0],
            "number_emergency": [1],
            "number_inpatient": [0]
        }

        df = pd.DataFrame(sample_data)
        self.assertEqual(df.shape, (1, 7))


if __name__ == '__main__':
    print("=" * 50)
    print("Running Prediction Tests...")
    print("=" * 50)

    unittest.main(verbosity=2)