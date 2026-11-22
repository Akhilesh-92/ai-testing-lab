import os
import requests
import pytest

from tests.rag_test_data import RAG_TEST_CASES


BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8002")


@pytest.mark.parametrize("test_data", RAG_TEST_CASES)
def test_retrieval_precision(test_data):

    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": test_data["question"]},
        timeout=180
    )

    assert response.status_code == 200

    result = response.json()

    sources = [
        item["source"]
        for item in result["sources"]
    ]

    relevant_chunks = sum(
        1
        for source in sources
        if source == test_data["expected_source"]
    )

    precision = relevant_chunks / len(sources) if sources else 0

    print(f"\nQuestion: {test_data['question']}")
    print(f"Expected source: {test_data['expected_source']}")
    print(f"Retrieved sources: {sources}")
    print(f"Retrieval precision: {precision:.2f}")

    assert sources, "No documents retrieved"

    assert test_data["expected_source"] in sources, (
        "Expected source was not retrieved"
    )
