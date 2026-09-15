from app.document_loader import load_pdf

documents = load_pdf("data/langchain-guide.pdf")

print("Number of documents:", len(documents))

print("\nFirst document:")
print(documents[0])