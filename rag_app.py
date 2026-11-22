import os
from pathlib import Path

import chromadb
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="AI Lab RAG API")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GENERATION_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"

chroma_client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = chroma_client.get_or_create_collection(
    name="ai_lab_knowledge"
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


def get_embedding(text: str):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def generate_answer(prompt: str):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "rag-api"
    }


@app.post("/ingest")
def ingest_documents():

    files = list(KNOWLEDGE_DIR.glob("*.md"))

    if not files:
        raise HTTPException(
            status_code=404,
            detail="No knowledge documents found"
        )

    documents = []
    ids = []
    metadatas = []
    embeddings = []

    for file in files:

        content = file.read_text(encoding="utf-8")

        # Simple paragraph-based chunking for this first experiment.
        chunks = [
            chunk.strip()
            for chunk in content.split("\n\n")
            if chunk.strip()
        ]

        for index, chunk in enumerate(chunks):

            documents.append(chunk)
            ids.append(f"{file.stem}-{index}")
            metadatas.append({
                "source": file.name,
                "chunk_index": index
            })

            embeddings.append(get_embedding(chunk))

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return {
        "status": "success",
        "documents_processed": len(files),
        "chunks_indexed": len(documents)
    }


@app.post("/query")
def query_knowledge(request: QueryRequest):

    if collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is empty. Run /ingest first."
        )

    try:
        question_embedding = get_embedding(request.question)

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(3, collection.count())
        )

        retrieved_documents = results["documents"][0]
        retrieved_metadatas = results["metadatas"][0]

        context = "\n\n".join(retrieved_documents)

        prompt = f"""
You are a technical assistant.

Answer the question using ONLY the supplied context.

If the context does not contain the answer, clearly say:
"I don't have enough information in the knowledge base to answer this."

Do not invent facts.

Context:
{context}

Question:
{request.question}

Answer:
"""

        answer = generate_answer(prompt)

        return {
            "question": request.question,
            "answer": answer,
            "retrieved_context": retrieved_documents,
            "sources": retrieved_metadatas
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {exc}"
        )
