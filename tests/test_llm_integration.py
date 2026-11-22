import os
import time
import pytest
import requests

BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8001"
)


@pytest.mark.llm
@pytest.mark.parametrize(
    "prompt",
    [
        "Explain Kubernetes pods in simple terms.",
        "What is Docker?",
        "Explain the purpose of Kubernetes services."
    ]
)
def test_real_llm_response(prompt):

    start_time = time.time()

    response = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": prompt},
        timeout=180
    )

    duration = time.time() - start_time

    assert response.status_code == 200

    result = response.json()

    assert result["model"] == "qwen2.5:1.5b"
    assert result["prompt"] == prompt
    assert isinstance(result["response"], str)
    assert result["response"].strip() != ""

    print(f"\nPrompt: {prompt}")
    print(f"Response: {result['response']}")
    print(f"Response time: {duration:.2f} seconds")
