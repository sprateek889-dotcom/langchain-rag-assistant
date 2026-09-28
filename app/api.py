from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="An API for asking questions about your documents.",
    version="1.0.0"
)


# Request model
class AskRequest(BaseModel):
    question: str


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


@app.post("/ask")
def ask_question(request: AskRequest):
    return {
        "question": request.question,
        "message": "Question received successfully"
    }