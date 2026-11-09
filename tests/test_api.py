from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "running"}


def test_empty_question_rejected():
    response = client.post(
        "/questions/ask",
        json={"question": ""}
    )

    assert response.status_code == 422


def test_missing_question_rejected():
    response = client.post(
        "/questions/ask",
        json={}
    )

    assert response.status_code == 422


def test_unsupported_upload_type_rejected():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.docx", b"test content")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only PDF and TXT files are supported."
    )