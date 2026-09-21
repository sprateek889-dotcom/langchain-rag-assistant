from app.document_loader import load_pdf
from app.text_splitter import split_documents
from app.retriever import create_retriever
from app.rag_chain import create_rag_chain
from app.question_rewriter import create_question_rewriter


def create_rag_system():

    # 1. Load documents
    documents = load_pdf(
        "data/langchain-guide.pdf"
    )

    # 2. Split documents
    chunks = split_documents(documents)

    # 3. Create retriever
    retriever = create_retriever(chunks)

    # 4. Create question rewriter
    question_rewriter = create_question_rewriter()

    # 5. Create RAG chain
    rag_chain = create_rag_chain()

    return retriever, question_rewriter, rag_chain