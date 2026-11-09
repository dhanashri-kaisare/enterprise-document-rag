from fastapi import FastAPI
from app.question_routes import router as question_router
from app.document_routes import router as document_router

app = FastAPI(title="Enterprise Document RAG System")

app.include_router(question_router)
app.include_router(document_router)


@app.get("/health")
def health_check():
    return {"status": "running"}