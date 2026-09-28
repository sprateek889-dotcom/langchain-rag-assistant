from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

from app.question_rewriter import create_question_rewriter


load_dotenv()


def main():

    rewriter = create_question_rewriter()

    chat_history = [
        HumanMessage(
            content="What is retrieval augmented generation?"
        ),
        AIMessage(
            content="RAG combines document retrieval with LLM generation."
        )
    ]

    question = "What are its benefits?"

    rewritten_question = rewriter.invoke({
        "chat_history": chat_history,
        "question": question
    })

    print("\nOriginal question:")
    print(question)

    print("\nRewritten question:")
    print(rewritten_question)


if __name__ == "__main__":
    main()