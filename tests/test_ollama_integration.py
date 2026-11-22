import time
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


@pytest.mark.parametrize(
    "prompt",
    [
        "Explain Kubernetes pods in simple terms.",
        "What is Docker?",
        "Explain the purpose of a Kubernetes service."
    ]
)
def test_real_llm_response(prompt):

    start = time.time()

    response = client.post(
        "/generate",
        json={"prompt": prompt}
    )

    duration = time.time() - start

    assert response.status_code == 200

    result = response.json()

    assert result["response"].strip() != ""
    assert result["model"] == "qwen2.5:1.5b"

    print(f"\nPrompt: {prompt}")
    print(f"Response: {result['response']}")
    print(f"Duration: {duration:.2f} seconds")
