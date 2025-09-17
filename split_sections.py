"""Split vendor manual markdown into sections using the table of contents."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

__all__ = ["split_sections", "export_sections"]

_DEFAULT_DELIMITER = "====="
_TOC_PAGE_LIMIT = 20

_DOTTED_LEADER_RE = re.compile(r"[.·]{2,}")
_TOC_ENTRY_START_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s+")
_ENTRY_SPLIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.*)$")
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)+\s+\S")


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

    path = Path(markdown_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".md":
        raise ValueError(f"split_sections expects a .md file, got {path}")

    pages = _read_pages(path)
    headings = _extract_toc_entries(pages, max_pages=toc_pages)
    if not headings:
        raise ValueError("Failed to locate table-of-contents headings in the markdown file")
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

    sections = split_sections(args.markdown, delimiter=args.delimiter, toc_pages=args.toc_pages)
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
