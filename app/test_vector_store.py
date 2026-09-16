from app.document_loader import load_pdf
from app.text_splitter import split_documents
from app.vector_store import create_vector_store

# 1. Load PDF
documents = load_pdf(
    "data/langchain-guide.pdf"
)

print("Documents:", len(documents))

# 2. Split documents into chunks
chunks = split_documents(documents)

print("Chunks:", len(chunks))

# 3. Create vector store
vector_store = create_vector_store(chunks)

print("Vector store created successfully!")

# 4. Perform semantic search
query = "What is retrieval augmented generation?"

results = vector_store.similarity_search(
    query,
    k=3
)

# 5. Display search results
print("\nSearch Results:")

for i, result in enumerate(results):

    print("\n" + "=" * 60)
    print(f"RESULT {i + 1}")
    print("=" * 60)

    print(result.page_content)

    print("\nMetadata:")
    print(result.metadata)