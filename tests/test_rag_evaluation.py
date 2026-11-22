import os
import requests
import pytest

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)

from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase


BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8002")

judge_model = OllamaModel(
    model="qwen2.5:1.5b",
    base_url="http://localhost:11434",
    temperature=0
)

test_data = [
    "What is a Kubernetes Pod?",
    "What is the difference between a Docker image and container?"
]


@pytest.mark.parametrize("question", test_data)
def test_rag_quality(question):

    # Step 1: Call the RAG application
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": question},
        timeout=180
    )

    assert response.status_code == 200

    result = response.json()

    answer = result["answer"]
    retrieved_context = result["retrieved_context"]

    assert answer.strip()
    assert retrieved_context

    # Step 2: Create the DeepEval test case
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieved_context
    )

    # Step 3: Define RAG evaluation metrics
    metrics = [
        ContextualRelevancyMetric(
            threshold=0.5,
            model=judge_model,
            include_reason=True,
            async_mode=False
        ),
        FaithfulnessMetric(
            threshold=0.5,
            model=judge_model,
            include_reason=True,
            async_mode=False
        ),
        AnswerRelevancyMetric(
            threshold=0.5,
            model=judge_model,
            include_reason=True,
            async_mode=False
        )
    ]

    # Step 4: Execute evaluation
    for metric in metrics:

        metric.measure(test_case)

        print(f"\nQuestion: {question}")
        print(f"Metric: {metric.__class__.__name__}")
        print(f"Score: {metric.score}")
        print(f"Reason: {metric.reason}")

        assert metric.score is not None
