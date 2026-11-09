from pathlib import Path
from pypdf import PdfReader


def load_document(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8")

    elif path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))

        extracted_pages = []

        for page in reader.pages:
            extracted_text = page.extract_text()

            if extracted_text:
                extracted_pages.append(extracted_text)

        text = "\n".join(extracted_pages)

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not text.strip():
        raise ValueError(
            "No readable text found. The document may be empty "
            "or contain scanned pages that require OCR."
        )

    return text