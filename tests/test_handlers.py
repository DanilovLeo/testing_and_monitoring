import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

from ml_service.app import app

VALID_PAYLOAD = {
    "race": "White",
    "sex": "Male",
    "native.country": "United-States",
    "occupation": "Prof-specialty",
    "education": "Bachelors",
    "capital.gain": 0,
}

NEEDED_COLUMNS = ['race', 'sex', 'native.country', 'occupation', 'education', 'capital.gain']


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    model.feature_names_in_ = NEEDED_COLUMNS
    return model


@pytest.fixture
def client(mock_model):
    with patch('ml_service.app.MODEL') as mock_model_container:
        mock_model_container.get.return_value.model = mock_model
        mock_model_container.features = NEEDED_COLUMNS
        yield TestClient(app)


@pytest.fixture
def client_no_model():
    with patch('ml_service.app.MODEL') as mock_model_container:
        mock_model_container.get.return_value.model = None
        mock_model_container.features = NEEDED_COLUMNS
        yield TestClient(app)


def test_predict_valid(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_features(client):
    response = client.post("/predict", json={"race": "White"})
    assert response.status_code == 422


def test_predict_model_not_loaded(client_no_model):
    response = client_no_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_predict_invalid_json(client):
    response = client.post("/predict", content="not json",
                          headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
