from app.services.text_splitter import split_text


def test_empty_text():
    assert split_text("") == []
    assert split_text("   ") == []


def test_chunks_are_created():
    text = "This is a sentence. " * 100

    chunks = split_text(text)

    assert len(chunks) > 1


def test_custom_chunk_size():
    text = "This is a sentence. " * 100

    chunks = split_text(
        text,
        chunk_size=200,
        chunk_overlap=20
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_invalid_chunk_size():
    try:
        split_text("Hello world", chunk_size=0)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "chunk_size" in str(exc)


def test_invalid_overlap():
    try:
        split_text(
            "Hello world",
            chunk_size=100,
            chunk_overlap=100
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)

def test_adaptive_chunking_for_short_document():
    text = "Short document. " * 50

    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_adaptive_chunking_for_long_document():
    text = "Long document content. " * 1000

    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 800 for chunk in chunks)