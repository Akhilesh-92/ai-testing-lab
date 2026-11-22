import os
import requests

BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8001"
)


def test_health_endpoint():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10
    )

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_generate_endpoint_rejects_empty_prompt():
    response = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": ""},
        timeout=10
    )

    assert response.status_code == 422


def test_generate_endpoint_rejects_missing_prompt():
    response = requests.post(
        f"{BASE_URL}/generate",
        json={},
        timeout=10
    )

    assert response.status_code == 422
