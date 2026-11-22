import pytest

from deepeval.metrics import HallucinationMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase


judge_model = OllamaModel(
    model="qwen2.5:1.5b",
    base_url="http://localhost:11434",
    temperature=0
)

context = [
    (
        "Kubernetes is an open-source container orchestration platform. "
        "It automates deployment, scaling and management of "
        "containerized applications."
    )
]


test_data = [
    (
        "What does Kubernetes do?",
        "Kubernetes automates deployment, scaling and management "
        "of containerized applications."
    ),
    (
        "Who developed Kubernetes and when?",
        "Kubernetes was developed by Microsoft in 2020."
    )
]


@pytest.mark.parametrize("question,answer", test_data)
def test_hallucination_detection(question, answer):

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        context=context
    )

    metric = HallucinationMetric(
        threshold=0.5,
        model=judge_model,
        include_reason=True
    )

    metric.measure(test_case)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"Hallucination score: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert 0 <= metric.score <= 1
