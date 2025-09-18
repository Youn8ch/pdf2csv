from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    import pdfplumber
except ImportError:  # pragma: no cover - import guard for optional dependency
    pdfplumber = None  # type: ignore[assignment]


@dataclass
class SplitResult:
    """Result of splitting the PDF into table of contents and body pages."""

    toc_entries: List[str]
    toc_pages: List[Tuple[int, str]]
    content_pages: List[Tuple[int, str]]


_TOC_KEYWORDS = {
    "目录",
    "目 录",
    "CONTENTS",
    "Contents",
    "Table of Contents",
    "TABLE OF CONTENTS",
}


def _normalise_text(value: str) -> str:
    """Normalise whitespace and punctuation for reliable comparisons."""

    collapsed = " ".join(value.replace("\u3000", " ").split())
    collapsed = collapsed.replace("．", ".").replace("·", ".")
    return collapsed.strip()


def _normalise_digits(value: str) -> str:
    """Convert full-width numerals to ASCII digits while leaving others intact."""

    chars: List[str] = []
    for char in value:
        if "0" <= char <= "9":
            chars.append(char)
            continue
        try:
            digit = unicodedata.digit(char)
        except (TypeError, ValueError):
            chars.append(char)
        else:
            chars.append(str(digit))
    return "".join(chars)


def _page_has_toc_keyword(page_text: str) -> bool:
    """Return True if the first characters of a page indicate a TOC heading."""

    stripped = page_text.lstrip()
    if not stripped:
        return False

    window_raw = stripped[:10]
    window_normalised = _normalise_text(window_raw)
    window_compact = window_normalised.replace(" ", "")
    window_raw_cf = window_raw.casefold()
    window_normalised_cf = window_normalised.casefold()
    window_compact_cf = window_compact.casefold()

    if not window_compact:
        return False

    for keyword in _TOC_KEYWORDS:
        keyword_normalised = _normalise_text(keyword)
        keyword_compact = keyword_normalised.replace(" ", "")
        keyword_raw_cf = keyword.casefold()
        keyword_normalised_cf = keyword_normalised.casefold()
        keyword_compact_cf = keyword_compact.casefold()

        if keyword_raw_cf in window_raw_cf:
            return True
        if keyword_normalised_cf and keyword_normalised_cf in window_normalised_cf:
            return True
        if keyword_compact_cf and keyword_compact_cf in window_compact_cf:
            return True
        if keyword_compact_cf and keyword_compact_cf.startswith(window_compact_cf):
            return True
        if window_compact_cf and window_compact_cf.startswith(keyword_compact_cf):
            return True

    return False


def _line_looks_like_toc_entry(line: str) -> bool:
    """Heuristic to determine whether a line resembles a TOC entry."""

    cleaned = _normalise_text(_normalise_digits(line))
    if not cleaned or len(cleaned) < 4:
        return False
    if cleaned.replace(" ", "") in _TOC_KEYWORDS:
        return False

    match = re.search(r"(?P<page>\d+)$", cleaned)
    if not match:
        return False

    title = cleaned[: match.start()]
    title = title.rstrip(". ·•")
    title = title.strip()
    if len(title) < 2:
        return False

    # Require either dot leaders or double spacing/tabs to reduce false positives.
    has_alignment_hint = (
        "..." in line
        or ".." in line
        or "\t" in line
        or "  " in line
    )
    return has_alignment_hint


def _extract_toc_entries(page_text: str) -> List[str]:
    """Extract potential TOC entries from the given page text."""

    entries: List[str] = []
    seen: set[str] = set()

    for raw_line in page_text.splitlines():
        cleaned = _normalise_text(_normalise_digits(raw_line))
        if not cleaned:
            continue
        if cleaned.replace(" ", "") in _TOC_KEYWORDS:
            continue

        match = re.search(r"(?P<page>\d+)$", cleaned)
        if not match:
            continue

        page_number = match.group("page")
        title = cleaned[: match.start()]
        title = re.sub(r"[\.·•\s]+$", "", title).strip()
        if len(title) < 2:
            continue

        formatted = f"{title} ...... {page_number}"
        if formatted not in seen:
            entries.append(formatted)
            seen.add(formatted)

    return entries


def _is_toc_page(page_text: str, entries: List[str]) -> bool:
    """Decide whether a page represents part of the table of contents."""

    if not _page_has_toc_keyword(page_text):
        return False

    if entries:
        return True

    raw_lines = [line for line in page_text.splitlines() if line.strip()]
    if not raw_lines:
        return False

    normalised_lines = [_normalise_text(line) for line in raw_lines]

    for line in normalised_lines[:4]:
        if line.replace(" ", "") in _TOC_KEYWORDS:
            return True

    candidate_lines = [
        (raw, normalised)
        for raw, normalised in zip(raw_lines, normalised_lines)
        if re.search(r"\d", normalised)
    ]
    if len(candidate_lines) < 3:
        return False

    structured = sum(1 for raw, _ in candidate_lines if _line_looks_like_toc_entry(raw))
    return structured >= max(3, int(len(candidate_lines) * 0.5))


def split_sections(pages: List[str]) -> SplitResult:
    """Split pages into table-of-contents entries and body content.

    Pages before the detected TOC are discarded. The TOC is captured as a list
    of formatted strings, the raw TOC pages are preserved for inspection, and
    the remaining content is returned alongside the original (1-indexed) page
    numbers.
    """

    toc_entries: List[str] = []
    toc_pages: List[Tuple[int, str]] = []
    content_pages: List[Tuple[int, str]] = []
    toc_started = False
    in_toc = False

    for index, page_text in enumerate(pages, start=1):
        entries = _extract_toc_entries(page_text)
        is_toc = _is_toc_page(page_text, entries)

        if not toc_started:
            if is_toc:
                toc_started = True
                in_toc = True
                toc_pages.append((index, page_text))
                if entries:
                    toc_entries.extend(entries)
            else:
                continue
        elif in_toc:
            if is_toc:
                toc_pages.append((index, page_text))
                if entries:
                    toc_entries.extend(entries)
            else:
                in_toc = False
                if page_text.strip():
                    content_pages.append((index, page_text))
        else:
            content_pages.append((index, page_text))

    if not toc_started:
        content_pages = [(idx, text) for idx, text in enumerate(pages, start=1) if text.strip()]

    return SplitResult(
        toc_entries=toc_entries, toc_pages=toc_pages, content_pages=content_pages
    )


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


def _write_markdown(result: SplitResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        if result.toc_entries:
            fh.write("# Table of Contents\n\n")
            for entry in result.toc_entries:
                fh.write(f"- {entry}\n")
            fh.write("\n")

        for page_number, content in result.content_pages:
            fh.write(f"## Page {page_number}\n\n")
            fh.write(content)
            fh.write("\n\n")


def _write_toc_pages(result: SplitResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        if not result.toc_pages:
            fh.write("No table-of-contents pages detected.\n")
            return

        for page_number, content in result.toc_pages:
            fh.write(f"## Page {page_number}\n")
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

    split_result = split_sections(pages)

    if args.preview:
        if split_result.toc_entries:
            toc_divider = "=" * 20 + " Table of Contents " + "=" * 20
            print(toc_divider)
            for entry in split_result.toc_entries:
                print(entry)
            print("=" * len(toc_divider))

        for page_number, content in split_result.content_pages:
            divider = f"{'=' * 20} Page {page_number} {'=' * 20}"
            print(divider)
            print(content)
            print("=" * len(divider))

    output_path = args.output or args.pdf.with_suffix(".md")
    _write_markdown(split_result, output_path)

    toc_output = output_path.with_name(output_path.stem + "_目录页.txt")
    _write_toc_pages(split_result, toc_output)

    page_count = len(split_result.content_pages)
    print(
        f"Wrote {page_count} page{'s' if page_count != 1 else ''} of body text to {output_path}"
    )
    if split_result.toc_entries:
        toc_count = len(split_result.toc_entries)
        print(
            f"Captured {toc_count} TOC entr{'ies' if toc_count != 1 else 'y'} before the body."
        )
    print(f"TOC pages saved to {toc_output}")
    if args.max_pages is not None:
        print(
            "Note: extraction stopped after",
            f" {min(args.max_pages, len(pages))} page(s) as requested.",
        )


if __name__ == "__main__":
    main()
