import requests
import pytest

BASE_URL = "http://localhost:8002"


def test_rag_rejects_empty_question():
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": ""}
    )

    assert response.status_code == 422


def test_rag_rejects_missing_question():
    response = requests.post(
        f"{BASE_URL}/query",
        json={}
    )

    assert response.status_code == 422


def test_rag_handles_unrelated_question():
    question = "Who won the FIFA World Cup in 1986?"

    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": question},
        timeout=180
    )

    assert response.status_code == 200

    result = response.json()

    assert "answer" in result
    assert result["answer"].strip()
    assert "retrieved_context" in result
    assert "sources" in result

    print("\nQuestion:", question)
    print("Answer:", result["answer"])
    print("Retrieved context:", result["retrieved_context"])
