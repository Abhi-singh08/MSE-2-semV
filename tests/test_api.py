from fastapi.testclient import TestClient
from src.api import app

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

def test_health_endpoint_shape():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert "loaded_models" in response.json()

def test_predict_requires_models_or_returns_prediction():
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code in (422, 503)
