from langchain_chroma import Chroma
from app.config import (
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_DIRECTORY
)
from app.embeddings import create_embeddings

def create_vector_store(chunks):
    """
    Create a new vector store from document chunks.
    """

    embeddings = create_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=VECTOR_COLLECTION_NAME,
        persist_directory=VECTOR_DB_DIRECTORY
    )

    return vector_store


def load_vector_store():
    """
    Load an existing vector store from disk.
    """

    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=VECTOR_COLLECTION_NAME,
        persist_directory=VECTOR_DB_DIRECTORY,
        embedding_function=embeddings
    )

    return vector_store