from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import answer_question
from app.services.validation_service import validate_answer

router = APIRouter(prefix="/questions", tags=["Questions"])


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about uploaded documents")
    filename: Optional[str] = Field(default=None, description="Optional document filename to restrict search")


@router.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        result = answer_question(request.question, filename_filter=request.filename)

        if not result.get("context"):
            validation = {"is_supported": False, "validation": "NOT_SUPPORTED"}
        else:
            validation = validate_answer(context=result["context"], answer=result["answer"])

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "validation": validation,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to answer and validate the question.") from exc