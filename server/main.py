from fastapi import Request
import os
import shutil
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from load_dotenv import load_dotenv

from app.routes.route import router

# ─── Bootstrap ─────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    yield
    # ── Shutdown: auto-cleanup uploads & vector store ──
    logger.info("Server shutting down — running cleanup…")
    uploads_dir = "./uploads"
    if os.path.exists(uploads_dir):
        shutil.rmtree(uploads_dir)
        logger.info("Uploads directory removed.")
    try:
        from app.services.embedding import reset_vector_store
        reset_vector_store()
        logger.info("ChromaDB reset.")
    except Exception as e:
        logger.warning("ChromaDB reset failed during shutdown: %s", e)
    logger.info("Shutdown cleanup complete.")


app = FastAPI(
    title="AskThePDF API",
    description="Upload PDFs and ask questions — powered by LangChain RAG",
    version="0.1.0",
    lifespan=lifespan,
)

# ─── CORS (allow the frontend to call the API) ────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "AskThePDF API is running"}


@app.get("/health")
def health_check():
    checks = {}

    # Check ChromaDB
    try:
        from app.services.embedding import get_chroma_client
        client = get_chroma_client()
        client.heartbeat()
        checks["chromadb"] = True
    except Exception:
        checks["chromadb"] = False

    healthy = all(checks.values())

    return {
        "healthy": healthy,
        "status": "ok" if healthy else "Ill",
        "checks": checks
    }