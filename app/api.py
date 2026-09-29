import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from app.rag import create_rag_system
from app.schemas import AskRequest, AskResponse

from app.rag import create_rag_system

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="An API for asking questions about your documents.",
    version="1.0.0"
)

UPLOAD_DIRECTORY = Path("uploads")

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
    MAX_FILE_SIZE=10 * 1024 * 1024  # 10 MB
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

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document after validating its type and size.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )

    # Validate file extension
    safe_filename = Path(file.filename).name

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    try:
        # Read the uploaded file
        contents = await file.read()

        # Validate empty file
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Validate file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 10 MB limit."
            )

        # Validate basic PDF signature
        if not contents.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file does not appear to be a valid PDF."
            )

        # Save the validated file
        file_path = UPLOAD_DIRECTORY / safe_filename

        file_path.write_bytes(contents)

        return {
            "message": "PDF uploaded successfully.",
            "filename": safe_filename,
            "saved_to": str(file_path),
            "file_size_bytes": len(contents)
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Failed to save uploaded document.")

        raise HTTPException(
            status_code=500,
            detail="Failed to save the uploaded document."
        )

    finally:
        await file.close()

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

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected error while processing the question.")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question."
        )