import requests
import pytest
import os

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")


@pytest.mark.parametrize(
    "prompt",
    [
        "Explain Kubernetes in simple terms.",
        "What is Docker used for?",
        "Explain Pytest to a beginner.",
        "What is the difference between Docker and Kubernetes?",
    ]
)
def test_response_is_not_empty(prompt):

    response = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": prompt},
        timeout=180
    )

    assert response.status_code == 200

    answer = response.json()["response"]

    assert isinstance(answer, str)
    assert len(answer.strip()) > 20


@pytest.mark.parametrize(
    "prompt",
    [
        "What is the capital of Mars?",
        "Explain why 2 + 2 equals 5.",
        "Tell me the exact number of Kubernetes clusters running worldwide."
    ]
)
def test_uncertain_questions(prompt):

    response = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": prompt},
        timeout=180
    )

    assert response.status_code == 200

    answer = response.json()["response"]

    assert len(answer.strip()) > 10

    print(f"\nQuestion: {prompt}")
    print(f"AI response: {answer}")
