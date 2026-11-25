"""Unit and integration tests for Flask API endpoints."""

import json
import pytest
from src.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert "models" in data


def test_predict_endpoint_valid_human(client):
    text = (
        "When we examine the rise of the Roman Empire, historians often emphasize military superiority, "
        "yet civic administration and infrastructural cohesion were arguably far more decisive. "
        "Roman engineering—exemplified by aqueducts, paved highways, and fortified frontier outposts—allowed "
        "rapid communication and sustained economic integration across disparate provinces."
    )
    payload = {"text": text, "model": "baseline"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "label" in data
    assert data["label"] in ["Human", "AI"]
    assert "confidence" in data
    assert "probabilities" in data
    assert "stylometrics" in data
    assert "sentence_highlights" in data


def test_predict_endpoint_short_warning(client):
    payload = {"text": "This is very short text.", "model": "baseline"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["warning"] is not None
    assert "fewer than 50" in data["warning"] or "50 words" in data["warning"]


def test_predict_empty_text(client):
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_feedback_endpoint(client):
    payload = {
        "text": "Sample text for feedback test.",
        "predicted_label": "AI",
        "user_label": "Human",
        "model_used": "Baseline SVM",
        "notes": "Test feedback entry",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
