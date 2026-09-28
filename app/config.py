import os

from dotenv import load_dotenv

load_dotenv()

# LLM configuration
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-5-mini"
)

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0"
    )
)


# Embedding configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)


# Vector database configuration
VECTOR_DB_DIRECTORY = os.getenv(
    "VECTOR_DB_DIRECTORY",
    "chroma_db"
)

VECTOR_COLLECTION_NAME = os.getenv(
    "VECTOR_COLLECTION_NAME",
    "knowledge_base"
)

# Retrieval configuration
RETRIEVAL_K = int(
    os.getenv(
        "RETRIEVAL_K",
        "3"
    )
)