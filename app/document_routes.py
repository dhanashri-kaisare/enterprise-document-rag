from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.document_pipeline import process_document
from app.services.rag_service import reset_caches
from app.services.vector_store import delete_document_from_store, load_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    suffix = Path(file.filename).suffix.lower()
    saved_path = DATA_DIR / f"{uuid4().hex}{suffix}"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        saved_path.write_bytes(content)
        result = process_document(
            str(saved_path),
            original_filename=file.filename or saved_path.name
        )
        reset_caches()

        return {
            "message": "Document uploaded and processed successfully.",
            "filename": file.filename,
            "chunks_created": len(result["chunks"]),
            "characters_extracted": len(result["text"]),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc
    finally:
        await file.close()


@router.get("/list")
def list_documents():
    vector_store = load_vector_store()
    if vector_store is None:
        return {"documents": []}

    documents = {
        doc.metadata.get("filename")
        for doc in vector_store.docstore._dict.values()
        if doc.metadata.get("filename")
    }
    return {"documents": sorted(documents)}


@router.delete("/{filename}")
def delete_document(filename: str):
    deleted_count = delete_document_from_store(filename)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    reset_caches()
    return {
        "message": f"Deleted '{filename}' ({deleted_count} chunks removed).",
        "filename": filename,
        "chunks_deleted": deleted_count,
    }