

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability",
    version="1.0"
)

try:
    model = joblib.load("models/logistic_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# model = joblib.load("models/logistic_model.pkl")
# scaler = joblib.load("models/scaler.pkl")

FEATURE_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    
    Contract_One_year: int = Field(..., alias="Contract_One year")
    Contract_Two_year: int = Field(..., alias="Contract_Two year")
    InternetService_Fiber_optic: int = Field(..., alias="InternetService_Fiber optic")
    InternetService_No: int 
    OnlineSecurity_Yes: int
    TechSupport_Yes: int

    class Config:
        populate_by_name = True

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.post("/predict")
def predict_churn(data: CustomerData, threshold: float = 0.5):
    
    input_df = pd.DataFrame(0, columns=FEATURE_COLUMNS, index=[0])

    
    data_dict = data.dict(by_alias=True)

    for key, value in data_dict.items():
        if key in input_df.columns:
            input_df[key] = value

    input_scaled = scaler.transform(input_df)

    churn_prob = model.predict_proba(input_scaled)[0][1]
    churn_prediction = int(churn_prob >= threshold)

    return {
        "threshold": threshold,
        "churn_probability": round(float(churn_prob), 3),
        "churn_prediction": churn_prediction,
        "prediction_label": "Churn" if churn_prediction == 1 else "No Churn"
    }