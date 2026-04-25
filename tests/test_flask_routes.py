"""
Flask Route Tests for Healthcare Readmission Prediction Web Application.

Tests all routes using Flask's test_client() including edge cases.
"""

import unittest
import sys
import json
from pathlib import Path

# Add web directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "web"))


class TestFlaskRoutes(unittest.TestCase):
    """Test Flask routes using test_client()."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client."""
        from app import app
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test-secret-key'
        cls.client = app.test_client()
        cls.app = app

    # ---- GET / (Home Page) ----

    def test_home_page_returns_200(self):
        """Test that the home page loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_home_page_contains_form(self):
        """Test that the home page contains the prediction form."""
        response = self.client.get('/')
        html = response.data.decode('utf-8')
        self.assertIn('predictionForm', html)
        self.assertIn('age', html)
        self.assertIn('time_in_hospital', html)
        self.assertIn('Predict', html)

    def test_home_page_has_api_docs(self):
        """Test that the home page has real API documentation section."""
        response = self.client.get('/')
        html = response.data.decode('utf-8')
        self.assertIn('API Documentation', html)
        self.assertIn('/api/predict', html)
        # Should NOT have the fake alert() popup
        self.assertNotIn("alert('POST to /api/predict", html)

    def test_home_page_form_max_values(self):
        """Test that form max values match backend validation."""
        response = self.client.get('/')
        html = response.data.decode('utf-8')
        self.assertIn('max="365"', html)   # time_in_hospital
        self.assertIn('max="500"', html)   # num_lab_procedures
        self.assertIn('max="200"', html)   # num_medications

    # ---- POST /predict (Form Prediction) ----

    def test_predict_valid_data(self):
        """Test prediction with valid patient data."""
        response = self.client.post('/predict', data={
            'age': '65',
            'time_in_hospital': '4',
            'num_lab_procedures': '40',
            'num_medications': '10',
            'number_outpatient': '0',
            'number_emergency': '1',
            'number_inpatient': '0'
        })
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        # Should show either "Readmitted" or "Not Readmitted"
        self.assertTrue('Readmitted' in html or 'Not Readmitted' in html)

    def test_predict_shows_confidence(self):
        """Test that prediction result shows confidence score."""
        response = self.client.post('/predict', data={
            'age': '65',
            'time_in_hospital': '4',
            'num_lab_procedures': '40',
            'num_medications': '10',
            'number_outpatient': '0',
            'number_emergency': '1',
            'number_inpatient': '0'
        })
        html = response.data.decode('utf-8')
        self.assertIn('Confidence', html)

    def test_predict_shows_patient_info(self):
        """Test that result page displays submitted patient information."""
        response = self.client.post('/predict', data={
            'age': '72',
            'time_in_hospital': '7',
            'num_lab_procedures': '50',
            'num_medications': '15',
            'number_outpatient': '2',
            'number_emergency': '3',
            'number_inpatient': '1'
        })
        html = response.data.decode('utf-8')
        self.assertIn('72', html)  # Age should be displayed
        self.assertIn('Patient Information', html)

    def test_predict_missing_field(self):
        """Test prediction with missing required field."""
        response = self.client.post('/predict', data={
            'age': '65',
            # Missing other fields
        })
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        # Should show error page
        self.assertTrue('Error' in html or 'error' in html or 'Wrong' in html)

    # ---- POST /api/predict (JSON API) ----

    def test_api_predict_valid_json(self):
        """Test API prediction with valid JSON data."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 65,
                'time_in_hospital': 4,
                'num_lab_procedures': 40,
                'num_medications': 10,
                'number_outpatient': 0,
                'number_emergency': 1,
                'number_inpatient': 0
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('prediction_text', data)
        self.assertIn('confidence', data)
        self.assertIn(data['prediction'], [0, 1])

    def test_api_predict_missing_field(self):
        """Test API prediction with missing required field."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 65,
                # Missing other fields
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_api_predict_invalid_range(self):
        """Test API prediction with out-of-range values."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 200,  # Over max (120)
                'time_in_hospital': 4,
                'num_lab_procedures': 40,
                'num_medications': 10,
                'number_outpatient': 0,
                'number_emergency': 1,
                'number_inpatient': 0
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_api_predict_no_json(self):
        """Test API prediction with no JSON body."""
        response = self.client.post('/api/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        # Empty dict has no required fields, so should fail validation
        self.assertEqual(response.status_code, 400)

    def test_api_predict_invalid_json(self):
        """Test API prediction with invalid JSON (malformed body)."""
        response = self.client.post('/api/predict',
            data='not valid json',
            content_type='application/json'
        )
        # Flask may return 400 or 500 for malformed JSON depending on version
        self.assertIn(response.status_code, [400, 500])

    # ---- GET /history ----

    def test_history_page_returns_200(self):
        """Test that the history page loads successfully."""
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)

    def test_history_page_empty_initially(self):
        """Test that history page shows empty state initially."""
        response = self.client.get('/history')
        html = response.data.decode('utf-8')
        self.assertIn('No predictions yet', html)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for prediction inputs."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client."""
        from app import app
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test-secret-key'
        cls.client = app.test_client()

    def test_all_zeros_input(self):
        """Test prediction when all input values are 0."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 0,
                'time_in_hospital': 0,
                'num_lab_procedures': 0,
                'num_medications': 0,
                'number_outpatient': 0,
                'number_emergency': 0,
                'number_inpatient': 0
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn(data['prediction'], [0, 1])
        self.assertIsNotNone(data['confidence'])

    def test_maximum_values_input(self):
        """Test prediction with maximum allowed values."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 120,
                'time_in_hospital': 365,
                'num_lab_procedures': 500,
                'num_medications': 200,
                'number_outpatient': 100,
                'number_emergency': 50,
                'number_inpatient': 50
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn(data['prediction'], [0, 1])

    def test_boundary_values(self):
        """Test prediction at exact boundary values (min edge)."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 0,
                'time_in_hospital': 0,
                'num_lab_procedures': 0,
                'num_medications': 0,
                'number_outpatient': 0,
                'number_emergency': 0,
                'number_inpatient': 0
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)

    def test_negative_values_rejected(self):
        """Test that negative input values are rejected."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': -5,
                'time_in_hospital': 4,
                'num_lab_procedures': 40,
                'num_medications': 10,
                'number_outpatient': 0,
                'number_emergency': 1,
                'number_inpatient': 0
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_string_values_rejected(self):
        """Test that non-numeric string values are rejected."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 'sixty-five',
                'time_in_hospital': 4,
                'num_lab_procedures': 40,
                'num_medications': 10,
                'number_outpatient': 0,
                'number_emergency': 1,
                'number_inpatient': 0
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_float_values_accepted(self):
        """Test that floating point values are accepted."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 65.5,
                'time_in_hospital': 4.2,
                'num_lab_procedures': 40.0,
                'num_medications': 10.7,
                'number_outpatient': 0.0,
                'number_emergency': 1.0,
                'number_inpatient': 0.0
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)

    def test_typical_elderly_patient(self):
        """Test prediction for a typical high-risk elderly patient."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 85,
                'time_in_hospital': 12,
                'num_lab_procedures': 80,
                'num_medications': 25,
                'number_outpatient': 5,
                'number_emergency': 3,
                'number_inpatient': 4
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn(data['prediction'], [0, 1])

    def test_typical_young_patient(self):
        """Test prediction for a typical low-risk young patient."""
        response = self.client.post('/api/predict',
            data=json.dumps({
                'age': 25,
                'time_in_hospital': 1,
                'num_lab_procedures': 10,
                'num_medications': 2,
                'number_outpatient': 0,
                'number_emergency': 0,
                'number_inpatient': 0
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn(data['prediction'], [0, 1])


class TestPredictionHistory(unittest.TestCase):
    """Test session-based prediction history."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client."""
        from app import app
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test-secret-key'
        cls.app = app

    def test_prediction_stored_in_history(self):
        """Test that a prediction is stored in session history."""
        with self.app.test_client() as client:
            # Make a prediction
            client.post('/predict', data={
                'age': '65',
                'time_in_hospital': '4',
                'num_lab_procedures': '40',
                'num_medications': '10',
                'number_outpatient': '0',
                'number_emergency': '1',
                'number_inpatient': '0'
            })

            # Check history page
            response = client.get('/history')
            html = response.data.decode('utf-8')
            self.assertIn('65', html)  # Age should appear in history


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING FLASK ROUTE & EDGE CASE TESTS")
    print("=" * 60)
    unittest.main(verbosity=2)
