from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model

DEFAULT_SYSTEM_PROMPT = """You are a helpful PDF Q&A assistant. Follow these rules:

1. Answer ONLY using the provided context below. Never use outside knowledge.
2. Synthesize information across all context chunks to give a complete answer.
3. Use bullet points or numbered lists when the answer has multiple items.
4. Only say you don't have enough information if the context genuinely contains nothing relevant.
5. When referencing specific content, cite the page number if available.
6. Never repeat the question. Never add disclaimers or filler text.

Context from PDF:
{context}"""


def get_model():
    return init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        temperature=0.3,
    )


def build_prompt(system_prompt: str | None = None) -> ChatPromptTemplate:
    """Build a chat prompt template with an optional custom system prompt."""
    system = (system_prompt or DEFAULT_SYSTEM_PROMPT)
    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
