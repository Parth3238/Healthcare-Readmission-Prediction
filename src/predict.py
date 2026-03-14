import joblib
import pandas as pd
from pathlib import Path

# Get the project root directory (parent of src)
BASE_DIR = Path(__file__).parent.parent

# load model
model = joblib.load(BASE_DIR / "models" / "model.pkl")

# sample patient data (ALL features)
sample = pd.DataFrame({
    "age":[65],
    "time_in_hospital":[4],
    "num_lab_procedures":[40],
    "num_medications":[10],
    "number_outpatient":[0],
    "number_emergency":[1],
    "number_inpatient":[0]
})

prediction = model.predict(sample)

print("Prediction:", prediction)

