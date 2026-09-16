from app.document_loader import load_pdf

documents = load_pdf("data/langchain-guide.pdf")

print("Number of documents:", len(documents))

for i, document in enumerate(documents[:3]):

    print("\n" + "=" * 50)
    print(f"PAGE {i + 1}")
    print("=" * 50)
    print(document.page_content[:100])