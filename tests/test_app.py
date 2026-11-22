from fastapi.testclient import TestClient
import app

client = TestClient(app.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_generate(monkeypatch):

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "Kubernetes orchestrates containers."}

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(app.requests, "post", mock_post)

    response = client.post(
        "/generate",
        json={"prompt": "Explain Kubernetes"}
    )

    assert response.status_code == 200
    assert "Kubernetes" in response.json()["response"]


def test_empty_prompt():

    response = client.post(
        "/generate",
        json={"prompt": ""}
    )

    assert response.status_code == 422

