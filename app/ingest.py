from dotenv import load_dotenv

from app.document_loader import load_pdf
from app.text_splitter import split_documents
from app.vector_store import create_vector_store


load_dotenv()


def main():

    print("Starting document ingestion...")

    # 1. Load documents
    documents = load_pdf(
        "data/langchain-guide.pdf"
    )

    print(f"Loaded {len(documents)} document pages.")

    # 2. Split documents
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # 3. Create vector database
    create_vector_store(chunks)

    print("Vector database created successfully!")


if __name__ == "__main__":
    main()