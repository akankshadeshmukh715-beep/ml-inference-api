from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn ML API",
    description="Real-Time ML Inference REST API",
    version="1.0.0"
)

model = joblib.load("champion_model.joblib")


class PredictionRequest(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "message": "ML API is running"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = pd.DataFrame([request.model_dump()])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    return {
        "prediction": str(prediction),
        "probabilities": {
            str(model.classes_[0]): float(probabilities[0]),
            str(model.classes_[1]): float(probabilities[1])
        }
    }
