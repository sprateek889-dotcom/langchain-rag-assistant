from langchain_openai import OpenAIEmbeddings

def create_embeddings():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    return embeddings