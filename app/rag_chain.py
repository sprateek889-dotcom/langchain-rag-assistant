from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from app.rag_prompt import rag_prompt

def create_rag_chain():
    model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0
    )

    parser = StrOutputParser()

    chain = rag_prompt | model | parser

    return chain