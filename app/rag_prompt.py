from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI knowledge assistant.

        Answer the user's question using only the provided context.

        Use the conversation history to understand follow-up questions.

        If the answer cannot be found in the provided context,
        say:

        "I don't know based on the provided documents."

        Do not make up information.

        Context:
        {context}
        """
    ),
    (
        "placeholder",
        "{chat_history}"
    ),
    (
        "human",
        "{question}"
    )
])