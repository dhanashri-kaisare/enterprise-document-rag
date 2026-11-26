import re
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.services.vector_store import load_vector_store

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions using provided documents.

Rules:
1. Answer using only the supplied context, recognizing common synonyms (e.g., "remote work" = "work-from-home", "vacation" = "paid leave").
2. Do not invent facts or assume policies not explicitly stated in the context.
3. If the context does not contain the answer, say:
   "I could not find sufficient information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
""")

NO_ANSWER = "I could not find sufficient information in the uploaded documents."

# Cosine similarity pre-filter for dense vector candidates
MIN_COSINE_SIMILARITY = 0.15
# Cross-encoder logit threshold (filters out irrelevant chunks even when lexical keywords overlap)
MIN_RERANK_SCORE = -7.5

_vector_store = None
_bm25_index = None
_bm25_docs = []
_reranker = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def reset_caches():
    """Call this after uploading or deleting documents so indexes stay fresh."""
    global _vector_store, _bm25_index, _bm25_docs
    _vector_store = None
    _bm25_index = None
    _bm25_docs = []


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = load_vector_store()
    return _vector_store


def get_bm25_index(vector_store):
    global _bm25_index, _bm25_docs
    if _bm25_index is None:
        all_docs = list(vector_store.docstore._dict.values())
        _bm25_docs = all_docs
        tokenized_corpus = [_tokenize(doc.page_content) for doc in all_docs]
        _bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    return _bm25_index, _bm25_docs


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def hybrid_retrieve_and_rerank(
    question: str,
    vector_store,
    top_k: int = 4,
    filename_filter: str | None = None,
):
    """
    1. Dense Vector Search (FAISS) + Lexical Keyword Search (BM25)
    2. Optional document-scoped metadata filtering
    3. Reciprocal Rank Fusion (RRF) + Cross-Encoder Reranking
    """
    # 1. Dense semantic search (fetch wider pool if filtering by filename)
    fetch_k = 20 if filename_filter else 8
    dense_results = vector_store.similarity_search_with_score(question, k=fetch_k)
    dense_candidates = [
        (doc, float(score))
        for doc, score in dense_results
        if float(score) >= MIN_COSINE_SIMILARITY
        and (not filename_filter or doc.metadata.get("filename") == filename_filter)
    ][:8]

    # 2. Sparse BM25 keyword search
    bm25_index, bm25_docs = get_bm25_index(vector_store)
    bm25_candidates = []
    if bm25_index and bm25_docs:
        scores = bm25_index.get_scores(_tokenize(question))
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        for idx in ranked_indices:
            if scores[idx] <= 0:
                break
            doc = bm25_docs[idx]
            if not filename_filter or doc.metadata.get("filename") == filename_filter:
                bm25_candidates.append((doc, float(scores[idx])))
                if len(bm25_candidates) >= 8:
                    break

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    doc_map = {}
    cosine_map = {}

    for rank, (doc, cos_score) in enumerate(dense_candidates):
        key = (doc.metadata.get("filename", "unknown"), doc.page_content)
        doc_map[key] = doc
        cosine_map[key] = cos_score
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60 + rank + 1))

    for rank, (doc, _) in enumerate(bm25_candidates):
        key = (doc.metadata.get("filename", "unknown"), doc.page_content)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60 + rank + 1))

    if not doc_map:
        return [], len(dense_results)

    fused_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:8]
    fused_docs = [doc_map[k] for k in fused_keys]

    # 4. Cross-Encoder Reranking
    reranker = get_reranker()
    pairs = [(question, doc.page_content) for doc in fused_docs]
    rerank_scores = reranker.predict(pairs)

    reranked = []
    for key, doc, r_score in zip(fused_keys, fused_docs, rerank_scores):
        if float(r_score) >= MIN_RERANK_SCORE:
            reranked.append((doc, cosine_map.get(key, 0.0), float(r_score)))

    reranked.sort(key=lambda item: item[2], reverse=True)
    return reranked[:top_k], len(dense_results)


def answer_question(question: str, filename_filter: str | None = None) -> dict:
    vector_store = get_vector_store()

    if vector_store is None:
        raise ValueError("No documents have been indexed yet.")

    retrieval_start = time.perf_counter()
    relevant_results, dense_found = hybrid_retrieve_and_rerank(
        question,
        vector_store,
        top_k=4,
        filename_filter=filename_filter,
    )
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    print(f"Hybrid retrieval + reranking latency: {retrieval_ms:.2f} ms")

    if not relevant_results:
        return {
            "answer": NO_ANSWER,
            "sources": [],
            "context": "",
            "retrieval": {
                "chunks_found": dense_found,
                "chunks_accepted": 0,
                "min_cosine_similarity": MIN_COSINE_SIMILARITY,
                "min_rerank_score": MIN_RERANK_SCORE,
            },
        }

    context_parts = []
    sources = []

    for document, cos_score, rerank_score in relevant_results:
        filename = document.metadata.get("filename", "unknown")

        context_parts.append(
            f"Source: {filename}\n"
            f"Content: {document.page_content}"
        )

        sources.append({
            "filename": filename,
            "content": document.page_content,
            "cosine_similarity": round(cos_score, 4),
            "rerank_score": round(rerank_score, 4),
        })

    context = "\n\n".join(context_parts)

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "answer": response.content,
        "sources": sources,
        "context": context,
        "retrieval": {
            "chunks_found": dense_found,
            "chunks_accepted": len(relevant_results),
            "min_cosine_similarity": MIN_COSINE_SIMILARITY,
            "min_rerank_score": MIN_RERANK_SCORE,
        },
    }