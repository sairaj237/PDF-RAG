import chromadb
from chromadb import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import Optional

from app.services.load_document import load_pdf_documents


CHROMA_DB_DIR = "./vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_CHROMA_SETTINGS = Settings(allow_reset=True, anonymized_telemetry=False)

_embeddings: Optional[HuggingFaceEmbeddings] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_vector_store: Optional[Chroma] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR, settings=_CHROMA_SETTINGS
        )
    return _chroma_client


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            client=get_chroma_client(),
            embedding_function=get_embeddings(),
        )
    return _vector_store


def store_pdf_embeddings(file_path: str):
    documents = load_pdf_documents(file_path)
    vs = get_vector_store()
    vs.add_documents(documents)
    return vs


def reset_vector_store():
    """Reset ChromaDB and clear the cached instances so they reinitialise cleanly."""
    global _chroma_client, _vector_store
    client = get_chroma_client()
    client.reset()
    # Clear cache so next call rebuilds against the fresh DB
    _chroma_client = None
    _vector_store = None
