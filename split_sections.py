"""Split vendor manual markdown into sections using the table of contents."""
from __future__ import annotations

import argparse
from collections import Counter
import math
import re
from pathlib import Path
from typing import Sequence

__all__ = ["split_sections", "export_sections"]

_DEFAULT_DELIMITER = "====="
_TOC_PAGE_LIMIT = 20

_DOTTED_LEADER_RE = re.compile(r"[.·]{2,}")
_TOC_ENTRY_START_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+")
_ENTRY_SPLIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*)$")
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+\S")
_REPEATED_WHITESPACE_RE = re.compile(r"\s+")
_ROMAN_NUMERAL_RE = re.compile(r"\b[ivxlcdm]+\b", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")


def _ensure_markdown_suffix(path: Path) -> Path:
    """Return ``path`` with a ``.md`` suffix, replacing any existing suffix."""

    if path.suffix.lower() == ".md":
        return path
    return path.with_suffix(".md")


def _read_pages(path: Path) -> list[list[str]]:
    """Return the document as a list of pages without ``## Page`` markers."""

    pages: list[list[str]] = []
    current: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("## Page "):
                if current:
                    pages.append(current)
                    current = []
                continue
            current.append(line)
    if current:
        pages.append(current)
    return pages


def _normalize_repeated_line(text: str) -> str:
    """Return a normalised representation for header/footer comparisons."""

    collapsed = text.replace("\u3000", " ").strip()
    if not collapsed:
        return ""
    collapsed = _REPEATED_WHITESPACE_RE.sub(" ", collapsed).lower()
    collapsed = _DIGITS_RE.sub("<num>", collapsed)
    collapsed = _ROMAN_NUMERAL_RE.sub("<roman>", collapsed)
    return collapsed


def _take_nonempty(lines: Sequence[str], limit: int, *, from_end: bool = False) -> list[str]:
    """Return up to ``limit`` non-empty lines from the start or end of ``lines``."""

    result: list[str] = []
    iterable = reversed(lines) if from_end else iter(lines)
    for line in iterable:
        if line.strip():
            result.append(line)
            if len(result) >= limit:
                break
    if from_end:
        result.reverse()
    return result


def _collect_repeated_header_footer_patterns(
    pages: Sequence[Sequence[str]],
    *,
    top_lines: int = 5,
    bottom_lines: int = 5,
    min_ratio: float = 0.7,
    min_repeats: int = 2,
) -> tuple[set[str], set[str]]:
    """Return normalised header/footer line patterns that repeat across pages."""

    total_pages = len(pages)
    if total_pages < min_repeats:
        return set(), set()

    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    for page in pages:
        if not page:
            continue
        top_candidates = _take_nonempty(page, top_lines)
        bottom_candidates = _take_nonempty(page, bottom_lines, from_end=True)
        for candidate in top_candidates:
            normalized = _normalize_repeated_line(candidate)
            if normalized:
                top_counter[normalized] += 1
        for candidate in bottom_candidates:
            normalized = _normalize_repeated_line(candidate)
            if normalized:
                bottom_counter[normalized] += 1

    threshold = max(min_repeats, int(math.ceil(total_pages * min_ratio)))
    top_patterns = {pattern for pattern, count in top_counter.items() if count >= threshold}
    bottom_patterns = {pattern for pattern, count in bottom_counter.items() if count >= threshold}
    return top_patterns, bottom_patterns


def _trim_leading_blanks(lines: Sequence[str]) -> list[str]:
    """Remove leading blank lines from ``lines``."""

    trimmed = list(lines)
    while trimmed and not trimmed[0].strip():
        trimmed.pop(0)
    return trimmed


def _strip_repeated_patterns(
    lines: Sequence[str],
    patterns: set[str],
    *,
    from_end: bool = False,
) -> list[str]:
    """Remove header/footer lines whose normalised form is present in ``patterns``."""

    if not patterns:
        return list(lines)

    trimmed = list(lines)
    while trimmed:
        index = -1 if from_end else 0
        candidate = trimmed[index]
        if not candidate.strip():
            trimmed.pop(index)
            continue
        normalized = _normalize_repeated_line(candidate)
        if normalized in patterns:
            trimmed.pop(index)
            continue
        break
    return trimmed


def _remove_repeated_headers_and_footers(pages: Sequence[Sequence[str]]) -> list[list[str]]:
    """Strip recurring header/footer blocks from each page."""

    top_patterns, bottom_patterns = _collect_repeated_header_footer_patterns(pages)
    if not top_patterns and not bottom_patterns:
        return [list(page) for page in pages]

    cleaned_pages: list[list[str]] = []
    for page in pages:
        lines = list(page)
        lines = _strip_repeated_patterns(lines, top_patterns)
        lines = _strip_repeated_patterns(lines, bottom_patterns, from_end=True)
        lines = _trim_leading_blanks(_trim_trailing_blanks(lines))
        cleaned_pages.append(lines)
    return cleaned_pages


def _normalize_marker(text: str) -> str:
    """Collapse whitespace in ``text`` for easier marker comparisons."""

    return re.sub(r"\s+", "", text.replace("\u3000", ""))


def _normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace characters in ``text``."""

    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def _is_toc_marker(text: str) -> bool:
    """Return ``True`` if ``text`` marks the beginning of the table of contents."""

    normalized = _normalize_marker(text)
    return normalized in {"目录", "目錄"}


def _clean_toc_entry(raw: str) -> str | None:
    """Normalise a raw table-of-contents entry into a heading string."""

    candidate = raw.replace("\u3000", " ")
    match = _ENTRY_SPLIT_RE.match(candidate)
    if not match:
        return None
    label, title = match.groups()
    title = re.sub(r"[.·]{2,}.*$", "", title)
    title = re.sub(r"\s+\d+\s*$", "", title)
    title = _normalize_spaces(title)
    title = re.sub(r"[:：]\s*$", "", title).rstrip()
    if not title:
        return None
    heading = f"{label} {title}"
    if not _NUMBERED_HEADING_RE.match(heading):
        return None
    return heading


def _extract_toc_entries(pages: Sequence[Sequence[str]], *, max_pages: int = _TOC_PAGE_LIMIT) -> list[str]:
    """Return headings discovered in the table of contents."""

    entries: list[str] = []
    seen: set[str] = set()
    buffer: list[str] = []
    toc_started = False

    for page_index, page in enumerate(pages):
        if page_index >= max_pages:
            break
        for line in page:
            stripped = line.strip()
            if not stripped:
                continue
            if not toc_started:
                if _is_toc_marker(stripped):
                    toc_started = True
                continue
            if _is_toc_marker(stripped):
                continue
            if _TOC_ENTRY_START_RE.match(stripped) and _DOTTED_LEADER_RE.search(stripped):
                if buffer:
                    entry = _clean_toc_entry(" ".join(buffer))
                    if entry and entry not in seen:
                        entries.append(entry)
                        seen.add(entry)
                    buffer = []
                buffer = [stripped]
            elif buffer:
                buffer.append(stripped)
    if buffer:
        entry = _clean_toc_entry(" ".join(buffer))
        if entry and entry not in seen:
            entries.append(entry)
    return entries


def _normalize_content_heading(line: str) -> str | None:
    """Return a normalised heading from a document line."""

    candidate = line.replace("\u3000", " ").strip()
    match = _ENTRY_SPLIT_RE.match(candidate)
    if not match:
        return None
    label, title = match.groups()
    title = _normalize_spaces(title)
    title = re.sub(r"[:：]\s*$", "", title).rstrip()
    if not title:
        return None
    heading = f"{label} {title}"
    if not _NUMBERED_HEADING_RE.match(heading):
        return None
    return heading


def _trim_trailing_blanks(lines: Sequence[str]) -> list[str]:
    """Remove trailing blank lines from ``lines``."""

    trimmed = list(lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return trimmed


def _segment_by_headings(
    pages: Sequence[Sequence[str]],
    headings: Sequence[str],
    *,
    delimiter: str,
) -> list[str]:
    """Split the document into sections using the discovered headings."""

    if not headings:
        return []

    heading_to_index: dict[str, int] = {}
    for idx, heading in enumerate(headings):
        heading_to_index.setdefault(heading, idx)

    sections: list[str] = []
    current_lines: list[str] = []
    current_index = -1

    for page in pages:
        for line in page:
            stripped = line.strip()
            if not stripped and not current_lines:
                continue
            if stripped == delimiter:
                continue
            if not _DOTTED_LEADER_RE.search(stripped):
                normalized = _normalize_content_heading(line)
                if normalized is not None:
                    target_index = heading_to_index.get(normalized)
                    if target_index is not None and target_index > current_index:
                        if current_lines:
                            section = "\n".join(_trim_trailing_blanks(current_lines))
                            if section.strip():
                                sections.append(section)
                        current_lines = [line.strip()]
                        current_index = target_index
                        continue
            if current_lines:
                current_lines.append(line.rstrip())
    if current_lines:
        section = "\n".join(_trim_trailing_blanks(current_lines))
        if section.strip():
            sections.append(section)
    return sections


def _load_pages_and_headings(
    markdown_path: str | Path,
    *,
    toc_pages: int,
) -> tuple[list[list[str]], list[str]]:
    """Return document pages and discovered headings for ``markdown_path``."""

    path = Path(markdown_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".md":
        raise ValueError(f"split_sections expects a .md file, got {path}")

    raw_pages = _read_pages(path)
    pages = _remove_repeated_headers_and_footers(raw_pages)
    headings = _extract_toc_entries(pages, max_pages=toc_pages)
    if not headings:
        raise ValueError("Failed to locate table-of-contents headings in the markdown file")
    return pages, headings


def export_sections(
    sections: Sequence[str],
    output_path: str | Path,
    *,
    delimiter: str = _DEFAULT_DELIMITER,
) -> None:
    """Write ``sections`` to ``output_path`` separated by ``delimiter`` lines."""

    path = _ensure_markdown_suffix(Path(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_sections = [section.strip("\n") for section in sections if section.strip()]
    content = f"\n{delimiter}\n".join(normalized_sections)
    if content and not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


def split_sections(
    markdown_path: str | Path,
    *,
    delimiter: str = _DEFAULT_DELIMITER,
    toc_pages: int = _TOC_PAGE_LIMIT,
) -> list[str]:
    """Split the markdown manual into sections based on table-of-contents entries."""

    pages, headings = _load_pages_and_headings(markdown_path, toc_pages=toc_pages)
    return _segment_by_headings(pages, headings, delimiter=delimiter)


def _preview(section: str, limit: int = 20) -> str:
    """Return a sample of ``section`` capped at ``limit`` lines."""

    lines = section.splitlines()
    if len(lines) <= limit:
        return "\n".join(lines)
    head = "\n".join(lines[:limit])
    return f"{head}\n… ({len(lines) - limit} more lines)"


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Split markdown into chapter sections")
    parser.add_argument("markdown", type=Path, help="Path to the markdown file generated from a PDF")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output markdown file (default: <input stem>_sections.md)",
    )
    parser.add_argument(
        "-n",
        "--preview",
        type=int,
        default=0,
        help="Print the first N sections after splitting",
    )
    parser.add_argument(
        "--print-toc",
        action="store_true",
        help="Only print the extracted table-of-contents headings and exit",
    )
    parser.add_argument(
        "--delimiter",
        default=_DEFAULT_DELIMITER,
        help="String used to separate sections when exporting (default: '=====')",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip writing the split sections to disk",
    )
    parser.add_argument(
        "--toc-pages",
        type=int,
        default=_TOC_PAGE_LIMIT,
        help="Number of initial pages to scan for the table of contents",
    )
    args = parser.parse_args(argv)

    pages, headings = _load_pages_and_headings(args.markdown, toc_pages=args.toc_pages)

    if args.print_toc:
        for heading in headings:
            print(heading)
        return

    sections = _segment_by_headings(pages, headings, delimiter=args.delimiter)
    print(f"Detected {len(sections)} sections in {args.markdown}")

    if not args.no_export:
        default_output = args.markdown.with_name(args.markdown.stem + "_sections.md")
        target = args.output or default_output
        export_sections(sections, target, delimiter=args.delimiter)
        normalized_target = _ensure_markdown_suffix(Path(target))
        print(f"Wrote sections to {normalized_target}")

    if args.preview:
        for index, section in enumerate(sections[: args.preview], start=1):
            print("\n" + args.delimiter)
            print(f"Section {index}")
            print(args.delimiter)
            print(_preview(section))


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
