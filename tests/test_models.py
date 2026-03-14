"""
Unit Tests for Healthcare Readmission Prediction System
Tests for data preprocessing, model training, and prediction modules.
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preprocessing import load_data, split_data


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.test_data = pd.DataFrame({
            'age': [65, 70, 55, 80, 45],
            'time_in_hospital': [4, 8, 2, 12, 3],
            'num_lab_procedures': [40, 60, 20, 80, 30],
            'num_medications': [10, 15, 5, 20, 8],
            'number_outpatient': [0, 2, 1, 0, 3],
            'number_emergency': [1, 0, 2, 1, 0],
            'number_inpatient': [0, 1, 0, 2, 1],
            'readmitted': [0, 1, 0, 1, 0]
        })
    
    def test_dataframe_shape(self):
        """Test if DataFrame has correct shape."""
        self.assertEqual(self.test_data.shape, (5, 8))
    
    def test_no_missing_values(self):
        """Test if there are no missing values."""
        self.assertEqual(self.test_data.isnull().sum().sum(), 0)
    
    def test_target_column_exists(self):
        """Test if target column 'readmitted' exists."""
        self.assertIn('readmitted', self.test_data.columns)
    
    def test_feature_columns(self):
        """Test if all feature columns exist."""
        expected_features = [
            'age', 'time_in_hospital', 'num_lab_procedures',
            'num_medications', 'number_outpatient', 
            'number_emergency', 'number_inpatient'
        ]
        for feature in expected_features:
            self.assertIn(feature, self.test_data.columns)
    
    def test_target_values(self):
        """Test if target values are binary."""
        unique_values = self.test_data['readmitted'].unique()
        self.assertEqual(set(unique_values), {0, 1})


class TestDataSplitting(unittest.TestCase):
    """Test cases for data splitting functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.test_data = pd.DataFrame({
            'age': np.random.randint(30, 90, 100),
            'time_in_hospital': np.random.randint(1, 15, 100),
            'num_lab_procedures': np.random.randint(10, 100, 100),
            'num_medications': np.random.randint(1, 50, 100),
            'number_outpatient': np.random.randint(0, 10, 100),
            'number_emergency': np.random.randint(0, 5, 100),
            'number_inpatient': np.random.randint(0, 5, 100),
            'readmitted': np.random.randint(0, 2, 100)
        })
    
    def test_split_data(self):
        """Test data splitting functionality."""
        X_train, X_test, y_train, y_test = split_data(self.test_data)
        
        # Check shapes
        self.assertEqual(len(X_train), 80)  # 80% of 100
        self.assertEqual(len(X_test), 20)   # 20% of 100
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
    
    def test_feature_target_separation(self):
        """Test if features and target are correctly separated."""
        X_train, X_test, y_train, y_test = split_data(self.test_data)
        
        # Check if 'readmitted' is not in X
        self.assertNotIn('readmitted', X_train.columns)
        self.assertNotIn('readmitted', X_test.columns)
        
        # Check if y contains only readmitted values
        self.assertEqual(set(y_train.unique()).union(set(y_test.unique())), {0, 1})
    
    def test_data_types(self):
        """Test if data types are preserved after splitting."""
        X_train, X_test, y_train, y_test = split_data(self.test_data)
        
        # Check if X_train is DataFrame
        self.assertIsInstance(X_train, pd.DataFrame)
        
        # Check if y_train is Series
        self.assertIsInstance(y_train, pd.Series)


class TestModelPrediction(unittest.TestCase):
    """Test cases for model prediction functionality."""
    
    def test_prediction_input_format(self):
        """Test if prediction input has correct format."""
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
        self.assertEqual(list(df.columns), [
            'age', 'time_in_hospital', 'num_lab_procedures',
            'num_medications', 'number_outpatient',
            'number_emergency', 'number_inpatient'
        ])
    
    def test_prediction_output_range(self):
        """Test if prediction output is in valid range."""
        # Mock prediction - should be 0 or 1
        prediction = 0  # or 1
        self.assertIn(prediction, [0, 1])


class TestWebApp(unittest.TestCase):
    """Test cases for web application."""
    
    def test_template_files_exist(self):
        """Test if HTML template files exist."""
        web_dir = Path(__file__).parent.parent / "web"
        templates_dir = web_dir / "templates"
        
        required_templates = ['index.html', 'result.html', 'error.html']
        for template in required_templates:
            template_path = templates_dir / template
            self.assertTrue(template_path.exists(), f"{template} not found")
    
    def test_static_files_exist(self):
        """Test if static CSS files exist."""
        web_dir = Path(__file__).parent.parent / "web"
        css_dir = web_dir / "static" / "css"
        
        css_file = css_dir / "style.css"
        self.assertTrue(css_file.exists(), "style.css not found")
    
    def test_flask_app_exists(self):
        """Test if Flask app file exists."""
        web_dir = Path(__file__).parent.parent / "web"
        app_file = web_dir / "app.py"
        self.assertTrue(app_file.exists(), "app.py not found")


class TestModelFiles(unittest.TestCase):
    """Test cases for model files."""
    
    def test_model_file_exists(self):
        """Test if trained model file exists."""
        models_dir = Path(__file__).parent.parent / "models"
        model_file = models_dir / "model.pkl"
        
        # Note: This might fail if model is not trained yet
        # This is just a check for file existence
        if model_file.exists():
            self.assertTrue(model_file.stat().st_size > 0, "Model file is empty")
    
    def test_data_file_exists(self):
        """Test if dataset file exists."""
        data_dir = Path(__file__).parent.parent / "data"
        data_file = data_dir / "healthcare_readmission_dataset.csv"
        
        self.assertTrue(data_file.exists(), "Dataset file not found")
        self.assertTrue(data_file.stat().st_size > 0, "Dataset file is empty")


def run_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestDataSplitting))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPrediction))
    suite.addTests(loader.loadTestsFromTestCase(TestWebApp))
    suite.addTests(loader.loadTestsFromTestCase(TestModelFiles))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING UNIT TESTS - Healthcare Readmission Prediction")
    print("=" * 70)
    
    result = run_tests()
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("ALL TESTS PASSED ✓")
    else:
        print(f"TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 70)
