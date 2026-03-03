import sys
import os

# Ajouter le dossier courant au PYTHONPATH
sys.path.append(os.getcwd())

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    payload = {"text": "This product is amazing and works perfectly"}
    response = client.post("/predict", json=payload, headers={"x-api-key": "secret123"})
    assert response.status_code == 200
    data = response.json()
    assert "cluster_id" in data
    assert "theme" in data
    assert "confidence" in data


def test_train_not_available():
    response = client.post("/train")
    assert response.status_code == 404

def test_predict_unauthorized():
    payload = {"text": "Test message"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 401
