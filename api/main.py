
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            Path(__file__).resolve().parent.parent
            / "models"
            / "registered_model"
        )
    )
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained Logistic Regression model.",
    version="1.0.0"
)

model = None
model_load_error = None

try:
    model = mlflow.sklearn.load_model(str(MODEL_PATH))
    print("Local registered model loaded successfully.")

except Exception as error:
    model_load_error = str(error)
    print("Model loading failed:", model_load_error)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData] = Field(min_length=1)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {model_load_error}"
        )

    return {
        "status": "healthy",
        "model_name": "Customer-Churn-Logistic-Regression",
        "model_version": "1",
        "model_loaded": True
    }


def generate_predictions(customers):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {model_load_error}"
        )

    try:
        dataframe = pd.DataFrame(customers)

        predictions = model.predict(dataframe)
        probabilities = model.predict_proba(dataframe)

        results = []

        for prediction, probability in zip(predictions, probabilities):
            churn_probability = float(probability[1])

            results.append({
                "prediction": int(prediction),
                "churn_label": "Yes" if int(prediction) == 1 else "No",
                "churn_probability": round(churn_probability, 4),
                "retention_probability": round(
                    float(probability[0]), 4
                )
            })

        return results

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


@app.post("/predict")
def predict(customer: CustomerData):
    results = generate_predictions([customer.model_dump()])
    return results[0]


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    customers = [
        customer.model_dump()
        for customer in request.customers
    ]

    results = generate_predictions(customers)

    return {
        "total_customers": len(results),
        "predictions": results
    }
