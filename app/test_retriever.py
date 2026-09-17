from app.document_loader import load_pdf
from app.text_splitter import split_documents
from app.retriever import create_retriever

# 1. Load PDF
documents = load_pdf(
    "data/langchain-guide.pdf"
)

print("Documents:", len(documents))

# 2. Split documents
chunks = split_documents(documents)

print("Chunks:", len(chunks))


# 3. Create retriever
retriever = create_retriever(chunks)

print("Retriever created successfully!")


# 4. Ask a question
query = "What is retrieval augmented generation?"


# 5. Retrieve relevant chunks
results = retriever.invoke(query)


# 6. Display results
print("\nRetrieved Results:")

for i, result in enumerate(results):

    print("\n" + "=" * 60)
    print(f"RESULT {i + 1}")
    print("=" * 60)

    print(result.page_content)

    print("\nMetadata:")
    print(result.metadata)