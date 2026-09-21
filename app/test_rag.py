from dotenv import load_dotenv

from app import rag_chain
from app.rag import create_rag_system

load_dotenv()


def main():

    # Create RAG system
    retriever, rag_chain = create_rag_system()

    print("RAG system created successfully!")

    # Ask question
    question = input("\nYou: ")

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Combine document content
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Generate answer
    #response = rag_chain.invoke({
    #    "context": context,
    #    "question": question
    #})

    #print("\nAI:", response)

    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAI:", response)

    print("\nSources:")

    for document in documents:
        print(
            f"- Page {document.metadata.get('page', 'Unknown')}"
        )

if __name__ == "__main__":
    main()