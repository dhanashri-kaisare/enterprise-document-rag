---
title: Enterprise Document Rag
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false

---# Enterprise Document Retrieval & Context Validation System

An end-to-end Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **LangChain**, **FAISS**, **Hugging Face embeddings**, and **Groq (`openai/gpt-oss-20b`)**, designed to answer enterprise policy questions strictly from uploaded documents while preventing hallucinations.

## Key Features
- **Document Ingestion Pipeline:** Supports loading, chunking, embedding, and indexing enterprise documents into a local FAISS vector store.
- **Hallucination Guardrails:** Enforces cosine-similarity thresholding and strict context-only prompting to refuse unanswerable or out-of-scope queries.
- **Automated LLM Evaluation & CI/CD:** Runs a 25-case evaluation suite (`tests/test_rag_eval.py`) on every push via **GitHub Actions** using **pytest** and an **LLM-as-a-Judge** scorer to enforce an 85%+ quality gate.

## Evaluation Benchmarks
Evaluated across 25 test cases covering direct facts, paraphrased queries, nuance checks, and unanswerable hallucination traps:

| Iteration | Changes Applied | Pass Rate | Unanswerable Rejection | Avg Retrieval Latency |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Default `MIN_COSINE_SIMILARITY = 0.35`, `k = 3`, uncached vector store | 76% (19/25) | 100% (10/10) | ~40 ms |
| **Tuned** | Synonym-aware system prompt, calibrated threshold (`0.15`), `k = 5`, cached FAISS store | **92% (23/25)** | **100% (10/10)** | **~35 ms** |

## Project Structure
```text
├── app/
│   ├── services/          # Document loader, splitter, FAISS vector store, RAG & validation logic
│   ├── document_routes.py # Endpoints for document upload and indexing
│   ├── question_routes.py # Endpoints for grounded Q&A queries
│   └── main.py            # FastAPI application entrypoint
├── tests/
│   ├── eval_dataset.json  # 25-case ground-truth evaluation dataset
│   ├── test_rag_eval.py   # Pytest + LLM-as-a-Judge faithfulness & correctness assertions
│   └── run_eval.py        # CI quality-gate runner (>= 85% pass threshold)
└── .github/workflows/
    └── eval.yml           # Automated GitHub Actions evaluation pipeline

Local Setup
pip install -r requirements.txt
# Add GROQ_API_KEY=your_key to .env
uvicorn app.main:app --reload