import pytest
from fastapi.testclient import TestClient
import json
import os
from api.app import app
from config.settings import Config

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_openai_embedding():
    """Test the OpenAI embedding endpoint."""
    payload = {
        "text": "This is a test text for embedding.",
        "batch_size": Config.DEFAULT_BATCH_SIZE,
        "model": Config.OPENAI_MODEL,
        "embedding_provider": "openai",
        "embedding_model": Config.OPENAI_EMBEDDING_MODEL
    }
    response = client.post("/embed", json=payload)
    assert response.status_code == 200
    assert "embeddings" in response.json()