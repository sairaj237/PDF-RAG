"""
Basic RAG evaluation using LangSmith.

Steps:
    1. Start the FastAPI server: uv run uvicorn main:app --reload
    2. Upload your PDF via the Streamlit UI
    3. Run this script: uv run python evaluate.py

Results appear in LangSmith under the 'AskThePDF-Eval' experiment.
"""

import os
import requests
from load_dotenv import load_dotenv

load_dotenv()

from langsmith import Client, evaluate
from langsmith.schemas import Run, Example

ls = Client()
API_BASE = "http://localhost:8000/api"

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Edit these to match your uploaded PDF content

DATASET_NAME = "AskThePDF-QA-Research"

QA_PAIRS = [
    {
        "question": "What problem does federated learning solve for small datasets?",
        "expected": "Federated learning enables training on distributed small datasets across multiple clients without centralizing the data, addressing privacy and data scarcity issues.",
    },
    {
        "question": "What is FedAvg?",
        "expected": "FedAvg (Federated Averaging) is an algorithm that aggregates local model updates from clients by averaging them on the central server.",
    },
    {
        "question": "What are the main challenges of federated learning with small datasets?",
        "expected": "Key challenges include statistical heterogeneity, non-IID data distributions, communication overhead, and model convergence issues.",
    },
    {
        "question": "How does data heterogeneity affect federated learning?",
        "expected": "Non-IID (non-independent and identically distributed) data across clients causes model drift and degrades global model performance.",
    },
    {
        "question": "What techniques are used to improve federated learning on small datasets?",
        "expected": "Techniques include data augmentation, transfer learning, personalized federated learning, and regularization methods.",
    },
]


def ensure_dataset():
    existing = [d.name for d in ls.list_datasets()]
    if DATASET_NAME in existing:
        return ls.read_dataset(dataset_name=DATASET_NAME)

    dataset = ls.create_dataset(DATASET_NAME, description="AskThePDF RAG eval set")
    for pair in QA_PAIRS:
        ls.create_example(
            inputs={"question": pair["question"]},
            outputs={"answer": pair["expected"]},
            dataset_id=dataset.id,
        )
    print(f"Created dataset '{DATASET_NAME}' with {len(QA_PAIRS)} examples.")
    return dataset


import time

# ─── Target: calls the live API so the real vector store is used ──────────────

def rag_pipeline(inputs: dict) -> dict:
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={"message": inputs["question"], "session_id": "eval-session"},
            timeout=60,
        )
        resp.raise_for_status()
        time.sleep(4)  # stay under 15 RPM free tier limit
        return {"answer": resp.json().get("answer", "")}
    except Exception as e:
        print(f"  ✗ call failed: {e}")
        return {"answer": ""}


# ─── Evaluators ───────────────────────────────────────────────────────────────

def relevance_evaluator(run: Run, example: Example) -> dict:
    answer = (run.outputs or {}).get("answer", "").lower()
    expected = (example.outputs or {}).get("answer", "").lower()
    stop_words = {"a", "an", "the", "is", "are", "of", "in", "and", "that", "it"}
    expected_words = set(expected.split()) - stop_words
    matched = sum(1 for w in expected_words if w in answer)
    score = matched / len(expected_words) if expected_words else 0
    return {"key": "relevance", "score": round(score, 2)}


def grounded_evaluator(run: Run, example: Example) -> dict:
    answer = (run.outputs or {}).get("answer", "").lower().strip()
    not_grounded = (
        not answer
        or "i don't have enough information" in answer
        or "i cannot" in answer
        or "not mentioned" in answer
    )
    return {"key": "grounded", "score": 0 if not_grounded else 1}


# ─── Run evaluation ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sanity check — make sure the server is up and has data
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5)
        health.raise_for_status()
    except Exception:
        print("ERROR: FastAPI server is not running at http://localhost:8000")
        print("Start it first with: uv run uvicorn main:app --reload")
        exit(1)

    dataset = ensure_dataset()

    results = evaluate(
        rag_pipeline,
        data=DATASET_NAME,
        evaluators=[relevance_evaluator, grounded_evaluator],
        experiment_prefix="rag-eval",
        metadata={"model": "gemini-2.5-flash", "retriever": "mmr-k8"},
    )

    print("\nEvaluation complete. View results at https://smith.langchain.com")
