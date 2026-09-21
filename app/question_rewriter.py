from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a question rewriting assistant.

        Rewrite the user's latest question into a standalone question
        that can be understood without the conversation history.

        Preserve the original meaning.

        Do not answer the question.
        Only return the rewritten question.
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


def create_question_rewriter():

    model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0
    )

    parser = StrOutputParser()

    chain = rewrite_prompt | model | parser

    return chain