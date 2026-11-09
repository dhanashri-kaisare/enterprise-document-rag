from pathlib import Path

from app.services.document_loader import load_document
from app.services.text_splitter import split_text
from app.services.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
)


def process_document(file_path: str, original_filename: str | None = None):
    path = Path(file_path)
    filename = original_filename or path.name

    # Step 1: Extract text
    text = load_document(file_path)

    if not text.strip():
        raise ValueError("The document contains no extractable text.")

    # Step 2: Split into chunks
    chunks = split_text(text)

    if not chunks:
        raise ValueError("No text chunks were created.")

    # Step 3: Load existing index
    vector_store = load_vector_store()

    # Step 4: Create or update the index with filename metadata
    documents = [
        {
            "page_content": chunk,
            "metadata": {"filename": filename}
        }
        for chunk in chunks
    ]

    if vector_store is None:
        vector_store = create_vector_store(chunks, filename)
    else:
        vector_store.add_texts(
            texts=[doc["page_content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )

    # Step 5: Save the index
    save_vector_store(vector_store)

    return {
        "text": text,
        "chunks": chunks,
        "filename": filename,
        "vector_store": vector_store,
    }