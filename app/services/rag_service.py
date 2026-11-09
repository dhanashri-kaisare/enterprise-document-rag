from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import time

from app.services.vector_store import load_vector_store

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions using provided documents.

Rules:
1. Answer using only the supplied context.
2. Do not invent facts.
3. If the context does not contain the answer, say:
   "I could not find sufficient information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
""")

NO_ANSWER = (
    "I could not find sufficient information in the uploaded documents."
)

# Initial cosine-similarity threshold; calibrate using test documents.
MIN_COSINE_SIMILARITY = 0.35


def answer_question(question: str) -> dict:
    vector_store = load_vector_store()

    if vector_store is None:
        raise ValueError("No documents have been indexed yet.")

    # Measure document retrieval time
    retrieval_start = time.perf_counter()

    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    retrieval_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000

    print(f"Retrieval latency: {retrieval_ms:.2f} ms")

    # Filter results using the similarity threshold.
    # With normalized embeddings and MAX_INNER_PRODUCT,
    # the scores correspond to cosine similarity.
    relevant_results = [
        (document, float(score))
        for document, score in results
        if float(score) >= MIN_COSINE_SIMILARITY
    ]

    # Return early if no relevant document chunks are found.
    if not relevant_results:
        return {
            "answer": NO_ANSWER,
            "sources": [],
            "context": "",
            "retrieval": {
                "chunks_found": len(results),
                "chunks_accepted": 0,
                "min_cosine_similarity": MIN_COSINE_SIMILARITY
            }
        }

    context_parts = []
    sources = []

    for document, score in relevant_results:
        filename = document.metadata.get("filename", "unknown")

        context_parts.append(
            f"Source: {filename}\n"
            f"Content: {document.page_content}"
        )

        sources.append({
            "filename": filename,
            "content": document.page_content,
            "cosine_similarity": round(score, 4)
        })

    context = "\n\n".join(context_parts)

    # Generate an answer using the retrieved context.
    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": response.content,
        "sources": sources,
        "context": context,
        "retrieval": {
            "chunks_found": len(results),
            "chunks_accepted": len(relevant_results),
            "min_cosine_similarity": MIN_COSINE_SIMILARITY
        }
    }