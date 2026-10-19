from pathlib import Path
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

    if not chunks:
        raise ValueError(
            "Cannot create vector store: no document chunks were provided."
        )

    embeddings = create_embeddings()

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=VECTOR_COLLECTION_NAME,
            persist_directory=VECTOR_DB_DIRECTORY
        )

    except Exception as error:
        raise RuntimeError(
            "Failed to create the Chroma vector store."
        ) from error

    return vector_store


def load_vector_store():
    """
    Load an existing vector store from disk.
    """

    db_path = Path(VECTOR_DB_DIRECTORY)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Vector database not found at '{db_path}'. "
            "Run 'python ingest.py' first."
        )

    embeddings = create_embeddings()

    try:
        vector_store = Chroma(
            collection_name=VECTOR_COLLECTION_NAME,
            persist_directory=VECTOR_DB_DIRECTORY,
            embedding_function=embeddings
        )

    except Exception as error:
        raise RuntimeError(
            "Failed to load the Chroma vector store."
        ) from error

    return vector_store

def add_documents_to_vector_store(chunks):
    """
    Add new document chunks to the existing Chroma collection.
    """

    if not chunks:
        raise ValueError(
            "Cannot add documents: no chunks were provided."
        )

    # Load the existing persistent vector database
    vector_store = load_vector_store()

    try:
        # Generate embeddings and store the new chunks
        document_ids = vector_store.add_documents(chunks)

    except Exception as error:
        raise RuntimeError(
            "Failed to add documents to the Chroma vector store."
        ) from error

    return len(document_ids)