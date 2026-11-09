from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.services.rag_service import answer_question, NO_ANSWER


def test_no_documents_indexed():
    with patch(
        "app.services.rag_service.load_vector_store",
        return_value=None
    ):
        try:
            answer_question("What is RAG?")
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "No documents have been indexed" in str(exc)


def test_unsupported_question_returns_no_answer():
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="RAG combines retrieval and generation.",
                metadata={"filename": "sample.txt"}
            ),
            0.20
        )
    ]

    with patch(
        "app.services.rag_service.load_vector_store",
        return_value=mock_store
    ):
        result = answer_question("What is the company's annual revenue?")

    assert result["answer"] == NO_ANSWER
    assert result["sources"] == []
    assert result["retrieval"]["chunks_accepted"] == 0

def test_relevant_question_uses_llm():
    mock_store = MagicMock()

    mock_store.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="RAG combines document retrieval with generation.",
                metadata={"filename": "sample.txt"}
            ),
            0.80
        )
    ]

    mock_response = MagicMock()
    mock_response.content = (
        "RAG combines document retrieval with language model generation."
    )

    with (
        patch(
            "app.services.rag_service.load_vector_store",
            return_value=mock_store
        ),
        patch(
            "app.services.rag_service.prompt.__or__"
        ) as mock_chain_creation
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        mock_chain_creation.return_value = mock_chain

        result = answer_question("What is RAG?")

    assert "RAG combines" in result["answer"]
    assert result["sources"][0]["filename"] == "sample.txt"
    assert result["retrieval"]["chunks_accepted"] == 1