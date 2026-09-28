from fastapi import FastAPI

app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="An API for asking questions about your documents.",
    version="1.0.0"
)

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