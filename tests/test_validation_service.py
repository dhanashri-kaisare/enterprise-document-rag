from unittest.mock import MagicMock, patch

from app.services.validation_service import validate_answer


def run_validation_with_mock(mock_content):
    mock_response = MagicMock()
    mock_response.content = mock_content

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    with patch(
        "app.services.validation_service.validator_llm"
    ), patch(
        "app.services.validation_service.validation_prompt"
    ) as mock_prompt:

        mock_prompt.__or__.return_value = mock_chain

        result = validate_answer(
            context="Employees receive 18 days of paid leave per year.",
            answer="Employees receive 18 days of paid leave per year."
        )

    return result


def test_supported_answer():
    result = run_validation_with_mock("SUPPORTED")

    assert result["is_supported"] is True
    assert result["validation"] == "SUPPORTED"


def test_unsupported_answer():
    result = run_validation_with_mock("NOT_SUPPORTED")

    assert result["is_supported"] is False
    assert result["validation"] == "NOT_SUPPORTED"


def test_invalid_validator_response():
    result = run_validation_with_mock("MAYBE")

    assert result["is_supported"] is False
    assert result["validation"] == "INVALID_VALIDATOR_RESPONSE"