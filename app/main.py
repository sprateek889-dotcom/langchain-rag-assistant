from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from app.chains import create_qa_chain

load_dotenv()

def main():

    chain = create_qa_chain()
    chat_history = []

    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        print("\nAI:", response)

        chat_history.append(
            HumanMessage(content=question)
        )

        chat_history.append(
            AIMessage(content=response)
        )
if __name__ == "__main__":
    main()