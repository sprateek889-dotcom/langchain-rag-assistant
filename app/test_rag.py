from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from app.rag import create_rag_system

load_dotenv()

def main():
    # Create RAG system
    try:
        retriever, question_rewriter, rag_chain = create_rag_system()

    except Exception as error:
        print(f"Failed to initialize the RAG system: {error}")
        return

    print("RAG system created successfully!")
    print("Ask questions about your documents.")
    print("Type 'exit' or 'quit' to stop.")

    # Conversation history
    chat_history = []

    while True:
        # Get user question
        question = input("\nYou: ")

        # Exit condition
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Validate empty questions
        if not question.strip():
            print("Please enter a question.")
            continue

        try:
            # 1. Rewrite the question using conversation history
            standalone_question = question_rewriter.invoke({
                "chat_history": chat_history,
                "question": question
            })

            print("\nRewritten question:")
            print(standalone_question)

            # 2. Retrieve relevant documents
            documents = retriever.invoke(standalone_question)

            if not documents:
                print(
                    "\nAI: I couldn't find relevant information "
                    "in the documents."
                )
                continue

            # 3. Combine retrieved documents into context
            context = "\n\n".join(
                document.page_content
                for document in documents
            )

            # 4. Generate answer using RAG chain
            response = rag_chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": question
            })

            # 5. Display answer
            print("\nAI:", response)

            # 6. Display sources
            print("\nSources:")

            for document in documents:
                print(
                    f"- Page {document.metadata.get('page', 'Unknown')}"
                )

            # 7. Update conversation history after a successful answer
            chat_history.append(
                HumanMessage(content=question)
            )

            chat_history.append(
                AIMessage(content=response)
            )

        except Exception as error:
            print(
                "\nSomething went wrong while answering your question: "
                f"{error}"
            )
            print("Please try again.")


if __name__ == "__main__":
    main()