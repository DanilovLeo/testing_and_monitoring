import pytest
from fastapi.testclient import TestClient
from ml_service.app import app, MODEL
from ml_service.mlflow_utils import configure_mlflow

REAL_RUN_ID = "49dbed45ad1a4e889ab467482facbf00"

VALID_PAYLOAD = {
    "race": "White",
    "sex": "Male",
    "native.country": "United-States",
    "occupation": "Prof-specialty",
    "education": "Bachelors",
    "capital.gain": 0,
}


@pytest.fixture(scope="module")
def integration_client():
    configure_mlflow()
    MODEL.set(run_id=REAL_RUN_ID)
    yield TestClient(app)


def test_full_pipeline(integration_client):
    response = integration_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_health_with_model(integration_client):
    response = integration_client.get("/health")
    assert response.status_code == 200
    assert response.json()["run_id"] == REAL_RUN_ID


def test_missing_features_returns_422(integration_client):
    response = integration_client.post("/predict", json={"race": "White"})
    assert response.status_code == 422
