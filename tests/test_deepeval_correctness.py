import os
import requests
import pytest

from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

judge_model = OllamaModel(
    model="qwen2.5:1.5b",
    base_url="http://localhost:11434",
    temperature=0
)

test_data = [
    (
        "What is Kubernetes?",
        "Kubernetes is an open-source container orchestration platform "
        "used to deploy, scale and manage containerized applications."
    ),
    (
        "What is Docker?",
        "Docker is a platform used to build, package and run applications "
        "inside containers."
    ),
    (
        "What is Pytest?",
        "Pytest is a Python testing framework used to write and execute "
        "automated tests."
    )
]


@pytest.mark.parametrize("question,expected_answer", test_data)
def test_answer_correctness(question, expected_answer):

    response = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": question},
        timeout=180
    )

    assert response.status_code == 200

    actual_answer = response.json()["response"]

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer
    )

    metric = GEval(
        name="Answer Correctness",
        criteria=(
            "Evaluate whether the actual output is factually correct "
            "compared with the expected output. Check for factual errors, "
            "contradictions and missing important information. "
            "Do not reward an answer merely because it is relevant."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=0.7,
        model=judge_model
    )

    metric.measure(test_case)

    print(f"\nQuestion: {question}")
    print(f"Expected: {expected_answer}")
    print(f"Actual: {actual_answer}")
    print(f"Correctness score: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert metric.score >= 0.7, (
        f"Answer correctness too low: {metric.score}"
    )

