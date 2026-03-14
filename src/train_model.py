from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from data_preprocessing import load_data,split_data

# Get the project root directory (parent of src)
BASE_DIR = Path(__file__).parent.parent

df = load_data(BASE_DIR / "data" / "healthcare_readmission_dataset.csv")

X_train, X_test, y_train, y_test = split_data(df)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, BASE_DIR / "models" / "model.pkl")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))





