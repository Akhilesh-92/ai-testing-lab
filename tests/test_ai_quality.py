import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


@pytest.mark.parametrize(
    "prompt, expected_terms",
    [
        (
            "What is Kubernetes?",
            ["container", "orchestration", "cluster"]
        ),
        (
            "What is Docker?",
            ["container", "image"]
        ),
        (
            "What is Pytest?",
            ["test", "python"]
        )
    ]
)
def test_ai_response_quality(prompt, expected_terms):

    response = client.post(
        "/generate",
        json={"prompt": prompt}
    )

    assert response.status_code == 200

    answer = response.json()["response"].lower()

    assert any(
        term in answer for term in expected_terms
    ), f"None of the expected terms were found: {answer}"
