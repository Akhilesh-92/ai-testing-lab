import os
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="AI Lab API")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "http://localhost:11434"
)

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate")
def generate(request: PromptRequest):

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": request.prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        return {
            "model": MODEL,
            "prompt": request.prompt,
            "response": response.json()["response"]
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {exc}"
        )
