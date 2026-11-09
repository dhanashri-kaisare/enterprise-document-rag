from app.services.rag_service import answer_question

question = "What is RAG?"

result = answer_question(question)

print("\nQuestion:", question)
print("\nAnswer:", result["answer"])
print("\nRetrieved Sources:")

for source in result["sources"]:
    print("-", source)