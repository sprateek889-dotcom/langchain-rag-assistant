from langchain_openai import OpenAIEmbeddings
from app.config import EMBEDDING_MODEL

def create_embeddings():

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    return embeddings