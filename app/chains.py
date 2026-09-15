from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from app.prompts import prompt

def create_qa_chain():

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=100
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain