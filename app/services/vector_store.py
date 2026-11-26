from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_STORE_DIR = Path("data/faiss_index")


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vector_store(chunks: list[str], filename: str):
    embeddings = get_embeddings()
    metadatas = [{"filename": filename} for _ in chunks]

    return FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        distance_strategy="MAX_INNER_PRODUCT"
    )


def save_vector_store(vector_store):
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))


def load_vector_store():
    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        return None

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy="MAX_INNER_PRODUCT"
    )


def delete_document_from_store(filename: str) -> int:
    """Deletes all chunks belonging to `filename` from FAISS and saves the updated index."""
    vector_store = load_vector_store()
    if vector_store is None:
        return 0

    ids_to_delete = [
        doc_id
        for doc_id, doc in vector_store.docstore._dict.items()
        if doc.metadata.get("filename") == filename
    ]

    if not ids_to_delete:
        return 0

    vector_store.delete(ids_to_delete)
    save_vector_store(vector_store)
    return len(ids_to_delete)