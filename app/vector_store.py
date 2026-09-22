from langchain_chroma import Chroma
from app.embeddings import create_embeddings

COLLECTION_NAME = "knowledge_base"
PERSIST_DIRECTORY = "chroma_db"

def create_vector_store(chunks):

    embeddings = create_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_store