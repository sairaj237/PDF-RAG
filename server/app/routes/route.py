import os
import uuid
import shutil
import logging

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.models.user import ChatRequest, ChatResponse, SourceDocument
from app.services.embedding import store_pdf_embeddings, get_chroma_client, reset_vector_store
from app.services.chain import get_rag_chain
from app.services.retriever import get_retriever

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "./uploads"


# ─── In-memory chat history store (keyed by session_id) ───────────────────────

_chat_histories: dict[str, list] = {}


# ─── PDF Upload ────────────────────────────────────────────────────────────────


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, save it to disk, then embed its content
    into the ChromaDB vector store so it can be queried later.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are accepted.",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save the uploaded PDF to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("Failed to save uploaded file: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file.")

    # Generate embeddings and store in ChromaDB
    try:
        store_pdf_embeddings(file_path)
    except Exception as e:
        logger.error("Embedding failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=f"Failed to process the PDF: {e}")

    return {
        "filename": file.filename,
        "message": "PDF uploaded and embedded successfully. You can now ask questions about it.",
    }


# ─── Chat (RAG) ───────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(data: ChatRequest):
    """
    Send a question and get an answer grounded in the uploaded PDF(s).
    Uses the RAG chain: retriever → prompt → LLM → answer.
    Previous chat messages are passed as context for multi-turn conversation.
    """
    session_id = data.session_id or str(uuid.uuid4())

    # Retrieve chat history for this session
    chat_history = _chat_histories.get(session_id, [])

    try:
        rag_chain = get_rag_chain(data.system_prompt)
        answer = rag_chain.invoke({
            "question": data.message,
            "chat_history": chat_history,
        })
    except Exception as e:
        logger.error("RAG chain error: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    # Append this exchange to the session's chat history
    from langchain_core.messages import HumanMessage, AIMessage
    chat_history.append(HumanMessage(content=data.message))
    chat_history.append(AIMessage(content=answer))
    _chat_histories[session_id] = chat_history

    # Also retrieve the source documents for transparency
    try:
        retriever = get_retriever()
        docs = retriever.invoke(data.message)
        sources = [
            SourceDocument(
                content=doc.page_content[:500],  # truncate for response size
                page=doc.metadata.get("page"),
                source=doc.metadata.get("source"),
            )
            for doc in docs
        ]
    except Exception:
        sources = []

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources=sources,
    )


# ─── Clear (wipe uploads + vector DB + chat history on exit) ──────────────────


@router.delete("/clear")
async def clear_vectorstore():
    """
    Delete all embedded documents from ChromaDB, remove uploaded files,
    and clear chat history. Call this when the user exits the platform.
    """
    # 1. Reset ChromaDB (drops all collections)
    try:
        reset_vector_store()
    except Exception as e:
        logger.error("Failed to clear ChromaDB: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {e}")

    # 2. Remove uploaded files
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        logger.info("Uploads directory cleared.")

    # 3. Clear all chat histories
    _chat_histories.clear()
    logger.info("Chat histories cleared.")

    return {"message": "Vector store, uploads, and chat history cleared successfully."}


# ─── Health ────────────────────────────────────────────────────────────────────


@router.get("/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}
