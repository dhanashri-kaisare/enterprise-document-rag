from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
load_dotenv()

validator_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

validation_prompt = ChatPromptTemplate.from_template("""
You are a strict document-grounding validator.

Compare the answer with the supplied context.

Return ONLY one of these labels:
SUPPORTED
NOT_SUPPORTED

Rules:
- Return SUPPORTED only if the context supports the answer's factual claims.
- Return NOT_SUPPORTED if the answer contains unsupported claims.
- Do not use outside knowledge.

Context:
{context}

Answer:
{answer}

Validation:
""")


def validate_answer(context: str, answer: str) -> dict:
    chain = validation_prompt | validator_llm

    result = chain.invoke({
        "context": context,
        "answer": answer
    })

    verdict = result.content.strip().upper()

    # Accept only an exact supported verdict
    is_supported = verdict == "SUPPORTED"

    return {
        "is_supported": is_supported,
        "validation": verdict if verdict in {
            "SUPPORTED", "NOT_SUPPORTED"
        } else "INVALID_VALIDATOR_RESPONSE"
    }