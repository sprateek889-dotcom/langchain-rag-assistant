from langchain_chroma import Chroma
from app.embeddings import create_embeddings

def create_vector_store(chunks):

    embeddings = create_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="knowledge_base"
    )

    return vector_store