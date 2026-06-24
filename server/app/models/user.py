from pydantic import BaseModel, Field
from typing import Optional


class UserPrompt(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    message: str = Field(min_length=1, max_length=2000, description="The user's question")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier for conversation tracking")
    system_prompt: Optional[str] = Field(default=None, description="Optional custom system prompt to override the default")


class SourceDocument(BaseModel):
    """A retrieved source chunk returned alongside the answer."""
    content: str
    page: Optional[int] = None
    source: Optional[str] = None


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""
    answer: str
    session_id: Optional[str] = None
    sources: list[SourceDocument] = []