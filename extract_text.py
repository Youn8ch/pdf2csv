from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    import pdfplumber
except ImportError:  # pragma: no cover - import guard for optional dependency
    pdfplumber = None  # type: ignore[assignment]


def extract_text(
    pdf_path: str | Path,
    *,
    header_cutoff: float = 0.05,
    footer_cutoff: float = 0.95,
    x_tolerance: float = 3.0,
    y_tolerance: float = 3.0,
    max_pages: int | None = None,
) -> List[str]:
    """Extract text content from each page of a PDF file.

    Parameters
    ----------
    pdf_path: str or Path
        Path to the PDF file on disk.
    header_cutoff: float, optional
        Top boundary (as a fraction of the page height) below which text is
        treated as header and discarded. Set to ``0`` to keep the full page.
    footer_cutoff: float, optional
        Bottom boundary (as a fraction of the page height) above which text is
        treated as footer and discarded. Set to ``1`` to keep the full page.
    x_tolerance: float, optional
        Horizontal tolerance passed to :meth:`pdfplumber.page.Page.extract_words`
        when clustering words on each line.
    y_tolerance: float, optional
        Vertical tolerance passed to :meth:`pdfplumber.page.Page.extract_words`
        when grouping words into lines.
    max_pages: int, optional
        Limit the number of pages to parse, which is useful when debugging.

    Returns
    -------
    list[str]
        A list where each entry contains the filtered text for the
        corresponding page in the PDF.
    """
    if pdfplumber is None:
        raise RuntimeError(
            "pdfplumber is required for extract_text but is not available."
        )

    if not 0 <= header_cutoff <= 1:
        raise ValueError("header_cutoff must be between 0 and 1.")
    if not 0 <= footer_cutoff <= 1:
        raise ValueError("footer_cutoff must be between 0 and 1.")
    if header_cutoff >= footer_cutoff:
        raise ValueError("header_cutoff must be smaller than footer_cutoff.")
    if max_pages is not None and max_pages < 0:
        raise ValueError("max_pages must be non-negative.")

    path = Path(pdf_path)
    pages: List[str] = []

    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages):
            if max_pages is not None and index >= max_pages:
                break

            page_height = float(page.height) if page.height else None
            header_limit = (
                header_cutoff * page_height if page_height and page_height > 0 else None
            )
            footer_limit = (
                footer_cutoff * page_height if page_height and page_height > 0 else None
            )

            words = page.extract_words(
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                keep_blank_chars=False,
                use_text_flow=True,
            )

            filtered_words = []
            for word in words:
                top = float(word.get("top", 0.0))
                bottom = float(word.get("bottom", 0.0))

                if header_limit is not None and bottom <= header_limit:
                    continue
                if footer_limit is not None and top >= footer_limit:
                    continue
                filtered_words.append(word)

            lines: List[str] = []
            current_line_id = None
            current_words: List[str] = []

            for word in filtered_words:
                line_id = word.get("line_number")
                if line_id is None:
                    line_id = (None, round(float(word.get("top", 0.0)), 1))

                if line_id != current_line_id:
                    if current_words:
                        lines.append(" ".join(current_words))
                        current_words = []
                    current_line_id = line_id

                text = str(word.get("text", "")).strip()
                if text:
                    current_words.append(text)

            if current_words:
                lines.append(" ".join(current_words))

            pages.append("\n".join(lines))

    return pages


def _write_markdown(pages: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for index, content in enumerate(pages, start=1):
            fh.write(f"## Page {index}\n\n")
            fh.write(content)
            fh.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text from a PDF using pdfplumber with configurable header"
            " and footer removal."
        )
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file to parse.")
    parser.add_argument(
        "--header-cutoff",
        type=float,
        default=0.05,
        help=(
            "Top boundary as a fraction of page height below which text is"
            " considered header. Use 0 to disable header filtering."
        ),
    )
    parser.add_argument(
        "--footer-cutoff",
        type=float,
        default=0.95,
        help=(
            "Bottom boundary as a fraction of page height above which text is"
            " considered footer. Use 1 to disable footer filtering."
        ),
    )
    parser.add_argument(
        "--x-tolerance",
        type=float,
        default=3.0,
        help="Horizontal tolerance passed to pdfplumber when clustering words.",
    )
    parser.add_argument(
        "--y-tolerance",
        type=float,
        default=3.0,
        help="Vertical tolerance passed to pdfplumber when grouping words.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Only parse the first N pages (useful while debugging).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print extracted page text to stdout for quick inspection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the Markdown output (defaults to <PDF>.md).",
    )

    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    pages = extract_text(
        args.pdf,
        header_cutoff=args.header_cutoff,
        footer_cutoff=args.footer_cutoff,
        x_tolerance=args.x_tolerance,
        y_tolerance=args.y_tolerance,
        max_pages=args.max_pages,
    )

    if args.preview:
        for index, content in enumerate(pages, start=1):
            divider = f"{'=' * 20} Page {index} {'=' * 20}"
            print(divider)
            print(content)
            print("=" * len(divider))

    output_path = args.output or args.pdf.with_suffix(".md")
    _write_markdown(pages, output_path)

    page_count = len(pages)
    print(f"Wrote {page_count} page{'s' if page_count != 1 else ''} of text to {output_path}")
    if args.max_pages is not None:
        print(
            "Note: extraction stopped after"
            f" {min(args.max_pages, page_count)} page(s) as requested."
        )


if __name__ == "__main__":
    main()