from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int = 50
) -> list[str]:
    """
    Split text into chunks using an adaptive chunk size.

    Short documents use smaller chunks, while longer documents
    use larger chunks to reduce unnecessary fragmentation.
    """
    if not text or not text.strip():
        return []

    text_length = len(text)

    # Choose chunk size based on document length
    if chunk_size is None:
        if text_length <= 2000:
            chunk_size = 300
        elif text_length <= 10000:
            chunk_size = 500
        else:
            chunk_size = 800

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    return splitter.split_text(text)