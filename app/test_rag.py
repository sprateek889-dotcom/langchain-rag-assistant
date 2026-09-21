from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

from app.rag import create_rag_system


load_dotenv()


def main():

    # Create RAG system
    retriever, rag_chain = create_rag_system()

    print("RAG system created successfully!")
    print("Ask questions about your documents.")
    print("Type 'exit' to quit.")

    # Conversation history
    chat_history = []

    while True:

        question = input("\nYou: ")

        # Exit condition
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Retrieve relevant documents
        documents = retriever.invoke(question)

        # Combine retrieved documents
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        # Generate answer
        response = rag_chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question
        })

        print("\nAI:", response)

        # Display sources
        print("\nSources:")

        for document in documents:
            print(
                f"- Page {document.metadata.get('page', 'Unknown')}"
            )

        # Update conversation history
        chat_history.append(
            HumanMessage(content=question)
        )

        chat_history.append(
            AIMessage(content=response)
        )


if __name__ == "__main__":
    main()