from langchain_chroma import Chroma
from app.embeddings import create_embeddings


COLLECTION_NAME = "knowledge_base"
PERSIST_DIRECTORY = "chroma_db"


def create_vector_store(chunks):
    """
    Create a new vector store from document chunks.
    Used during document ingestion.
    """

    embeddings = create_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_store


def load_vector_store():
    """
    Load an existing vector store from disk.
    Used during querying.
    """

    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    return vector_store
