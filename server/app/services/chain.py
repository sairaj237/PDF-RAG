from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional

from app.services.llm import get_model, build_prompt
from app.services.retriever import get_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(system_prompt: Optional[str] = None):
    retriever = get_retriever()
    model = get_model()
    prompt = build_prompt(system_prompt)

    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )
