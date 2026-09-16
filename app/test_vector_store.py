from app.document_loader import load_pdf
from app.text_splitter import split_documents
from app.vector_store import create_vector_store

documents = load_pdf(
    "data/langchain-guide.pdf"
)

chunks = split_documents(documents)

print("Documents:", len(documents))
print("Chunks:", len(chunks))

vector_store = create_vector_store(chunks)

print("Vector store created successfully!")