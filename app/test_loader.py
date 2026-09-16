from app.document_loader import load_pdf
from app.text_splitter import split_documents

documents = load_pdf("data/langchain-guide.pdf")

print("Number of documents:", len(documents))

chunks = split_documents(documents)

print("Chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):

    print("\n" + "=" * 60)
    print(f"CHUNK {i + 1}")
    print("=" * 60)

    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)