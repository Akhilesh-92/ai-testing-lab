import os
import requests
import pytest

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

judge_model = OllamaModel(
    model="qwen2.5:1.5b",
    base_url="http://localhost:11434",
    temperature=0
)

BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8001"
)


@pytest.mark.parametrize(
    "question",
    [
        "What is Kubernetes?",
        "What is Docker?",
        "What is Pytest?"
    ]
)
def test_answer_relevancy(question):

    # Get actual response from Kubernetes-hosted application
    response = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": question},
        timeout=180
    )

    assert response.status_code == 200

    actual_answer = response.json()["response"]

    # Create DeepEval test case
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer
    )

    # Evaluate semantic relevance
    metric = AnswerRelevancyMetric(
        threshold=0.6,
        model=judge_model
    )

    metric.measure(test_case)

    print(f"\nQuestion: {question}")
    print(f"Answer: {actual_answer}")
    print(f"Relevancy score: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert metric.score >= 0.6, (
        f"Answer relevancy too low: {metric.score}"
    )

