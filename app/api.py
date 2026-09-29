from fastapi import FastAPI, HTTPException
from app.rag import create_rag_system
from app.schemas import AskRequest, AskResponse

from app.rag import create_rag_system


app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="An API for asking questions about your documents.",
    version="1.0.0"
)

# RAG components
retriever = None
question_rewriter = None
rag_chain = None


def get_rag_system():
    """
    Initialize the RAG system only when it is first needed.
    Reuse the initialized components for subsequent requests.
    """

    global retriever, question_rewriter, rag_chain

    if retriever is None:
        retriever, question_rewriter, rag_chain = create_rag_system()

    return retriever, question_rewriter, rag_chain


@app.get("/")
def root():
    return {
        "message": "Welcome to the RAG Knowledge Assistant API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "RAG Knowledge Assistant API is running"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Accept a question, retrieve relevant documents,
    and generate an answer using the RAG system.
    """

    # Validate the question
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # 1. Get the RAG components
        retriever, question_rewriter, rag_chain = get_rag_system()

        # 2. Rewrite the question
        standalone_question = question_rewriter.invoke({
            "chat_history": [],
            "question": request.question
        })

        # 3. Retrieve relevant documents
        documents = retriever.invoke(standalone_question)

        # 4. Combine retrieved documents into context
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        # 5. Generate the answer
        answer = rag_chain.invoke({
            "context": context,
            "chat_history": [],
            "question": request.question
        })

        # 6. Collect source information
        sources = []

        for document in documents:
            sources.append({
                "page": document.metadata.get("page", "Unknown"),
                "source": str(document.metadata.get("source", "Unknown"))
            })

        # 7. Return the response
        return AskResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process the question: {str(error)}"
        ) from error