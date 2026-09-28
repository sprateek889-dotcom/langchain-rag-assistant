from app.retriever import create_retriever
from app.rag_chain import create_rag_chain
from app.question_rewriter import create_question_rewriter


def create_rag_system():

    # 1. Load existing vector database
    retriever = create_retriever()

    # 2. Create question rewriter
    question_rewriter = create_question_rewriter()

    # 3. Create RAG chain
    rag_chain = create_rag_chain()

    return retriever, question_rewriter, rag_chain

