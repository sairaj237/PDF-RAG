from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from typing import Optional

from app.services.llm import get_model, build_prompt
from app.services.retriever import get_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(system_prompt: Optional[str] = None):
    retriever = get_retriever()
    model = get_model()
    prompt = build_prompt(system_prompt)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    # Wrap with a name so LangSmith groups each Q&A as one named trace
    return chain.with_config(RunnableConfig(run_name="AskThePDF-RAG"))
