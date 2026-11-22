import os
import requests
import pytest

from deepeval.metrics import ContextualRecallMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from tests.rag_test_data import RAG_TEST_CASES


BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8002")

judge_model = OllamaModel(
    model="qwen2.5:1.5b",
    base_url="http://localhost:11434",
    temperature=0
)


@pytest.mark.parametrize("test_data", RAG_TEST_CASES)
def test_contextual_recall(test_data):

    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": test_data["question"]},
        timeout=180
    )

    assert response.status_code == 200

    result = response.json()

    test_case = LLMTestCase(
        input=test_data["question"],
        actual_output=result["answer"],
        expected_output=test_data["expected_answer"],
        retrieval_context=result["retrieved_context"]
    )

    metric = ContextualRecallMetric(
        threshold=0.5,
        model=judge_model,
        include_reason=True,
        async_mode=False
    )

    metric.measure(test_case)

    print(f"\nQuestion: {test_data['question']}")
    print(f"Expected: {test_data['expected_answer']}")
    print(f"Retrieved: {result['retrieved_context']}")
    print(f"Recall score: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert metric.score is not None
