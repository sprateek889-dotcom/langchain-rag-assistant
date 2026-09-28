from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from app.config import (
    LLM_MODEL,
    LLM_TEMPERATURE
)

from app.rag_prompt import rag_prompt

def create_rag_chain():
    model = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

    parser = StrOutputParser()

    chain = (
        rag_prompt
        | model
        | parser
    )

    return chain