from app.services.embedding import get_vector_store


def get_retriever():
    return get_vector_store().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20},
    )
