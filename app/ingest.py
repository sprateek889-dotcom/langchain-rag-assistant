from dotenv import load_dotenv

from app.document_loader import load_pdf
from app.text_splitter import split_documents
from app.vector_store import create_vector_store


load_dotenv()


def main():
    print("Starting document ingestion...")

    try:
        # 1. Load documents
        documents = load_pdf(
            "data/langchain-guide.pdf"
        )

        print(f"Loaded {len(documents)} document pages.")

        # 2. Split documents
        chunks = split_documents(documents)

        if not chunks:
            raise ValueError(
                "Document splitting produced no chunks."
            )

        print(f"Created {len(chunks)} chunks.")

        # 3. Create vector database
        create_vector_store(chunks)

        print("Vector database created successfully!")

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nIngestion error: {error}")

    except Exception as error:
        print(f"\nUnexpected ingestion error: {error}")


if __name__ == "__main__":
    main()