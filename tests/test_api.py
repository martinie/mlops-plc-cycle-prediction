import pytest


def test_health_endpoint():
    from app.main import app

    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_predict_rejects_missing_fields():
    from app.main import app

    client = app.test_client()
    response = client.post("/predict", json={"cycle_ms": 1500})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing required fields"