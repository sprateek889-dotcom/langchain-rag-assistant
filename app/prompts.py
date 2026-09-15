from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI knowledge assistant.

        Answer the user's questions clearly and accurately.

        If you don't know the answer, say that you don't know.
        Do not make up information.
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