from app.services.vector_store import load_vector_store

vector_store = load_vector_store()

if vector_store is None:
    print("No saved FAISS index found.")
else:
    questions = [
        "What is RAG?",
        "What is the company's annual revenue?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        results = vector_store.similarity_search_with_score(
            question,
            k=3
        )

        for document, distance in results:
            print(f"Distance: {distance:.4f}")
            print(f"Content: {document.page_content}")