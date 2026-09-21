from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI knowledge assistant.

        Answer the user's question using only the provided context.

        If the answer cannot be found in the context, say:
        "I don't know based on the provided documents."

        Do not make up information.

        Context:
        {context}
        """
    ),
    (
        "human",
        "{question}"
    )
])