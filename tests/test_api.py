
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65
}


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["model_version"] == "1"


def test_single_prediction():
    response = client.post(
        "/predict",
        json=SAMPLE_CUSTOMER
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["churn_label"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["retention_probability"] <= 1


def test_batch_prediction():
    response = client.post(
        "/predict/batch",
        json={
            "customers": [
                SAMPLE_CUSTOMER,
                SAMPLE_CUSTOMER
            ]
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["total_customers"] == 2
    assert len(result["predictions"]) == 2


def test_invalid_customer():
    response = client.post(
        "/predict",
        json={"gender": "Female"}
    )

    assert response.status_code == 422
