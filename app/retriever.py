from app.vector_store import create_vector_store

def create_retriever(chunks):
    vector_store = create_vector_store(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever