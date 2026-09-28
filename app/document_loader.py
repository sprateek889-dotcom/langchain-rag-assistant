from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """
    Load pages from a PDF file.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"The provided path is not a file: {pdf_path}"
        )

    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

    except Exception as error:
        raise RuntimeError(
            f"Failed to load PDF: {pdf_path}"
        ) from error

    if not documents:
        raise ValueError(
            f"No content was extracted from PDF: {pdf_path}"
        )

    return documents