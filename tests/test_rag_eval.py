import json
import re
from pathlib import Path
import pytest
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.services.rag_service import answer_question

load_dotenv()

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    EVAL_CASES = json.load(f)

judge_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


def judge_response(question: str, expected: str, actual: str, context: str, answerable: bool) -> dict:
    """Uses an LLM-as-a-Judge to evaluate Faithfulness and Correctness."""
    judge_prompt = f"""You are a strict RAG evaluation judge. Score the AI's answer on two metrics (0 or 1):
1. faithfulness: 1 if every claim in Actual Answer is strictly supported by the Retrieved Context (no made-up facts). 0 if it hallucinates anything outside the context.
2. correctness:
   - If Answerable is True: 1 if Actual Answer matches the meaning of Expected Answer.
   - If Answerable is False: 1 if Actual Answer clearly states the information is not found / not mentioned in the context rather than guessing.

Question: {question}
Answerable from docs: {answerable}
Expected Answer: {expected}
Retrieved Context: {context}
Actual Answer: {actual}

Respond ONLY with valid JSON in this exact format and nothing else:
{{"faithfulness": 1, "correctness": 1, "reason": "short explanation"}}"""

    response = judge_llm.invoke(judge_prompt)
    raw_text = response.content.strip()
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    return json.loads(match.group(0))


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["id"] for c in EVAL_CASES])
def test_rag_evaluation(case):
    result = answer_question(case["question"])
    actual_answer = result.get("answer", "")
    context = str(result.get("context", ""))

    scores = judge_response(
        question=case["question"],
        expected=case["expected_answer"],
        actual=actual_answer,
        context=context,
        answerable=case["answerable"],
    )

    assert scores["faithfulness"] == 1, f"[{case['id']}] Hallucination detected: {scores['reason']} | Answer: {actual_answer}"
    assert scores["correctness"] == 1, f"[{case['id']}] Incorrect answer: {scores['reason']} | Answer: {actual_answer}"