from fastapi import FastAPI
import mlflow
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
# Set MLflow tracking URI (update to your MLflow server address if needed)
mlflow.set_tracking_uri("http://mlflow-server:5000")  # Change to your MLflow server URI
model = mlflow.sklearn.load_model("models:/healthcare_classification/Production")


class PatientData(BaseModel):
    Age: float
    Gender: str
    Blood_Type: str
    Billing_Amount: float
    Admission_Type: str
    Medication: str
    Test_Results: str


@app.post("/predict")
def predict(data: PatientData):
    # Prepare input for model (match training features)
    input_dict = data.model_dump()
    # Rename keys to match training columns
    input_dict["Blood Type"] = input_dict.pop("Blood_Type")
    input_dict["Billing Amount"] = input_dict.pop("Billing_Amount")
    input_dict["Admission Type"] = input_dict.pop("Admission_Type")
    input_dict["Test Results"] = input_dict.pop("Test_Results")
    df = pd.DataFrame([input_dict])
    # One-hot encode categorical features
    categorical = [
        "Gender",
        "Blood Type",
        "Admission Type",
        "Medication",
        "Test Results",
    ]
    numerical = ["Age", "Billing Amount"]
    df_encoded = pd.get_dummies(df, columns=categorical)
    # Get expected columns from training
    model_features = numerical + [
        col for col in df_encoded.columns if col not in numerical
    ]

    df_encoded = df_encoded[model_features]
    pred = model.predict(df)
    # save production data into csv
    df["target"] = pred[0]
    df.to_csv("data/healthcare_production.csv", index=False)

    return {
        "prediction": int(pred[0]),
        "meaning": "Cancer" if pred[0] == 1 else "Not Cancer",
    }
