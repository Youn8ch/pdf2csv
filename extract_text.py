from typing import List
from pathlib import Path

try:
    import PyPDF2
except Exception as e:
    PyPDF2 = None


def extract_text(pdf_path: str | Path) -> List[str]:
    """Extract text content from each page of a PDF file.

    Parameters
    ----------
    pdf_path: str or Path
        Path to the PDF file on disk.

    Returns
    -------
    list of str
        A list where each entry contains the text for the corresponding
        page in the PDF. If a page has no extractable text or the PyPDF2
        dependency is unavailable, an empty string will be returned for
        that page.
    """
    path = Path(pdf_path)
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is required for extract_text but is not available.")
    pages: List[str] = []
    with path.open("rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python extract_text.py <PDF_PATH>")
    pdf = sys.argv[1]
    pages = extract_text(pdf)
    for i, content in enumerate(pages, 1):
        print(f"--- Page {i} ---")
        print(content)
