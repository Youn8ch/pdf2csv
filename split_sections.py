"""Split vendor manual markdown into sections using the table of contents."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = ["split_sections", "export_sections", "TOCReport", "HeadingInfo", "TOCMatch"]

_DEFAULT_DELIMITER = "====="
_TOC_PAGE_LIMIT = 20

_DOTTED_LEADER_RE = re.compile(r"[.·]{2,}")
_TOC_ENTRY_START_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+")
_ENTRY_SPLIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*)$")
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+\S")
_MULTILINE_HEADING_LOOKAHEAD = 4
_SENTENCE_PUNCTUATION_RE = re.compile(r"[。！？；;,，、]")


@dataclass
class HeadingInfo:
    """Location metadata for a heading extracted from the document body."""

    text: str
    page_index: int
    line_index: int

    def location_label(self) -> str:
        """Return a human-friendly ``page/line`` label for debugging."""

        return f"page {self.page_index + 1}, line {self.line_index + 1}"


@dataclass
class TOCMatch:
    """Pair a table-of-contents heading with its document counterpart."""

    toc_heading: str
    document_heading: HeadingInfo


@dataclass
class TOCReport:
    """Verification summary for the extracted table of contents."""

    raw_headings: list[str]
    matches: list[TOCMatch]
    missing: list[str]
    duplicates: list[str]
    orphans: list[HeadingInfo]

    @property
    def validated(self) -> list[str]:
        """Return the canonical headings confirmed to exist in the body."""

        return [match.document_heading.text for match in self.matches]

    @property
    def total(self) -> int:
        """Return the number of raw headings detected in the table of contents."""

        return len(self.raw_headings)

    @property
    def validated_count(self) -> int:
        """Return how many headings were confirmed against the document body."""

        return len(self.matches)

    @property
    def has_issues(self) -> bool:
        """Return ``True`` when verification uncovered inconsistencies."""

        return bool(self.missing or self.duplicates or self.orphans)

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


def _normalize_heading_key(text: str) -> str:
    """Return a normalised key for matching headings across sources."""

    normalized = text.replace("\u3000", " ")
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = normalized.replace("：", ":")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return re.sub(r"\s+", "", normalized).lower()


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


def _looks_like_toc_entry(page: Sequence[str], line_index: int, *, window: int = 4) -> bool:
    """Return ``True`` if the numbered ``page[line_index]`` behaves like a TOC entry."""

    line = page[line_index]
    if _DOTTED_LEADER_RE.search(line):
        return True

    limit = min(len(page), line_index + 1 + max(window, 1))
    for idx in range(line_index + 1, limit):
        candidate = page[idx].strip()
        if not candidate:
            continue
        if _DOTTED_LEADER_RE.search(candidate):
            return True
        if _TOC_ENTRY_START_RE.match(candidate) or _TOC_NUMBER_ONLY_RE.match(candidate):
            break
    return False


def _extract_toc_entries(
    pages: Sequence[Sequence[str]], *, max_pages: int = _TOC_PAGE_LIMIT
) -> tuple[list[str], tuple[int, int] | None]:
    """Return headings discovered in the table of contents and the last TOC position."""

    entries: list[str] = []
    seen: set[str] = set()
    buffer: list[str] = []
    split_buffer: list[str] = []
    toc_started = False
    last_position: tuple[int, int] | None = None

    for page_index, page in enumerate(pages):
        if page_index >= max_pages:
            break
        for line_index, line in enumerate(page):
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
                buffer = [stripped]
                last_position = (page_index, line_index)
                if has_dotted_leader:
                    flush_buffer()
                continue

            if buffer:
                buffer.append(stripped)
                last_position = (page_index, line_index)
    if buffer:
        entry = _clean_toc_entry(" ".join(buffer))
        if entry and entry not in seen:
            entries.append(entry)
    return entries, last_position


def _normalize_content_heading(line: str) -> str | None:
    """Return a normalised heading from a document line."""

    candidate = line.replace("\u3000", " ").strip()
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


def _extract_heading_at(
    lines: Sequence[str],
    start_index: int,
    *,
    max_merge_lines: int = _MULTILINE_HEADING_LOOKAHEAD,
) -> tuple[str | None, int, list[str]]:
    """Return a heading detected at ``start_index`` along with consumed lines."""

    raw_line = lines[start_index]
    stripped = raw_line.strip()
    if not stripped:
        return None, 1, []

    buffer: list[str] = [stripped]
    captured: list[str] = [raw_line.rstrip()]

    best_heading: str | None = _normalize_content_heading(stripped)
    best_consumed = 1 if best_heading is not None else 0
    best_captured: list[str] = captured.copy() if best_heading is not None else []
    initial_heading = best_heading

    if best_heading is None and not _TOC_NUMBER_ONLY_RE.match(stripped):
        return None, 1, []

    lookahead = start_index + 1
    merges = 0

    while lookahead < len(lines) and merges < max_merge_lines:
        candidate_raw = lines[lookahead]
        candidate_stripped = candidate_raw.strip()
        if not candidate_stripped:
            captured.append(candidate_raw.rstrip())
            lookahead += 1
            merges += 1
            continue
        if _NUMBERED_HEADING_RE.match(candidate_stripped):
            break
        if _TOC_NUMBER_ONLY_RE.match(candidate_stripped):
            break
        allow_extension = initial_heading is None
        if not allow_extension:
            break
        if best_heading is not None and _SENTENCE_PUNCTUATION_RE.search(candidate_stripped):
            break
        if best_heading is not None and len(candidate_stripped) > 30:
            break
        buffer.append(candidate_stripped)
        captured.append(candidate_raw.rstrip())
        merged = " ".join(buffer)
        normalized = _normalize_content_heading(merged)
        merges += 1
        if normalized is not None:
            best_heading = normalized
            best_consumed = lookahead - start_index + 1
            best_captured = captured.copy()
        lookahead += 1

    if best_heading is not None:
        consumed = max(best_consumed, 1)
        return best_heading, consumed, best_captured

    return None, 1, []


def _collect_document_headings(
    pages: Sequence[Sequence[str]],
    *,
    start_after: tuple[int, int] | None,
) -> tuple[dict[str, HeadingInfo], list[str]]:
    """Return body headings after ``start_after`` as a mapping and ordered keys."""

    heading_map: dict[str, HeadingInfo] = {}
    order: list[str] = []

    for page_index, page in enumerate(pages):
        if start_after is not None:
            toc_page, toc_line = start_after
            if page_index < toc_page:
                continue
            line_start = toc_line + 1 if page_index == toc_page else 0
        else:
            line_start = 0
        line_index = line_start
        while line_index < len(page):
            normalized, consumed, _ = _extract_heading_at(page, line_index)
            if normalized is not None:
                key = _normalize_heading_key(normalized)
                if key not in heading_map:
                    heading_map[key] = HeadingInfo(normalized, page_index, line_index)
                    order.append(key)
                line_index += consumed
            else:
                line_index += 1
    return heading_map, order


def _verify_toc_against_body(
    toc_headings: Sequence[str],
    heading_map: dict[str, HeadingInfo],
    heading_order: Sequence[str],
) -> TOCReport:
    """Cross-check table-of-contents headings against document headings."""

    matches: list[TOCMatch] = []
    missing: list[str] = []
    duplicates: list[str] = []
    seen_keys: set[str] = set()

    for heading in toc_headings:
        key = _normalize_heading_key(heading)
        if key in seen_keys:
            duplicates.append(heading)
            continue
        seen_keys.add(key)
        info = heading_map.get(key)
        if info is None:
            missing.append(heading)
            continue
        matches.append(TOCMatch(heading, info))

    toc_key_set = set(_normalize_heading_key(h) for h in toc_headings)
    orphans = [heading_map[key] for key in heading_order if key not in toc_key_set]

    return TOCReport(
        raw_headings=list(toc_headings),
        matches=matches,
        missing=missing,
        duplicates=duplicates,
        orphans=orphans,
    )


def _print_toc_report(report: TOCReport, *, stream=sys.stderr, limit: int = 10) -> None:
    """Emit a short verification summary for ``report`` to ``stream``."""

    print(
        f"TOC entries detected: {report.total}; validated in body: {report.validated_count}",
        file=stream,
    )
    if report.missing:
        print("Missing in body:", file=stream)
        for heading in report.missing[:limit]:
            print(f"  - {heading}", file=stream)
        if len(report.missing) > limit:
            remaining = len(report.missing) - limit
            print(f"  … {remaining} more", file=stream)
    if report.duplicates:
        print("Duplicate entries in TOC:", file=stream)
        for heading in report.duplicates[:limit]:
            print(f"  - {heading}", file=stream)
        if len(report.duplicates) > limit:
            remaining = len(report.duplicates) - limit
            print(f"  … {remaining} more", file=stream)
    if report.orphans:
        print("Headings seen in body but missing from TOC:", file=stream)
        for info in report.orphans[:limit]:
            print(f"  - {info.text} ({info.location_label()})", file=stream)
        if len(report.orphans) > limit:
            remaining = len(report.orphans) - limit
            print(f"  … {remaining} more", file=stream)
        print("Hint: increase --toc-pages to scan additional pages if these headings belong in the TOC.", file=stream)


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
    heading_key_to_index: dict[str, int] = {}
    for idx, heading in enumerate(headings):
        heading_to_index.setdefault(heading, idx)
        heading_key_to_index.setdefault(_normalize_heading_key(heading), idx)

    sections: list[str] = []
    current_lines: list[str] = []
    current_index = -1

    for page in pages:
        line_index = 0
        while line_index < len(page):
            line = page[line_index]
            stripped = line.strip()
            if not stripped and not current_lines:
                line_index += 1
                continue
            if stripped == delimiter:
                line_index += 1
                continue

            normalized, consumed, captured = _extract_heading_at(page, line_index)
            if normalized is not None:
                if any(_DOTTED_LEADER_RE.search(value) for value in captured if value.strip()):
                    normalized = None
                else:
                    target_index = heading_to_index.get(normalized)
                    if target_index is None:
                        target_index = heading_key_to_index.get(_normalize_heading_key(normalized))
                    if target_index is not None and target_index > current_index:
                        if current_lines:
                            section = "\n".join(_trim_trailing_blanks(current_lines))
                            if section.strip():
                                sections.append(section)
                        heading_lines = [value.strip() for value in captured if value.strip()]
                        if not heading_lines:
                            heading_lines = [page[line_index].strip()]
                        current_lines = heading_lines.copy()
                        current_index = target_index
                        line_index += consumed
                        continue

            if current_lines:
                if consumed > 1:
                    for offset in range(consumed):
                        current_lines.append(page[line_index + offset].rstrip())
                else:
                    current_lines.append(line.rstrip())
            line_index += max(consumed, 1)
    if current_lines:
        section = "\n".join(_trim_trailing_blanks(current_lines))
        if section.strip():
            sections.append(section)
    return sections


def _load_pages_and_headings(
    markdown_path: str | Path,
    *,
    toc_pages: int,
) -> tuple[list[list[str]], TOCReport]:
    """Return document pages and a verified table-of-contents report."""

    path = Path(markdown_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".md":
        raise ValueError(f"split_sections expects a .md file, got {path}")

    pages = _read_pages(path)
    toc_headings, toc_last_position = _extract_toc_entries(pages, max_pages=toc_pages)
    if not toc_headings:
        raise ValueError("Failed to locate table-of-contents headings in the markdown file")
    heading_map, heading_order = _collect_document_headings(pages, start_after=toc_last_position)
    report = _verify_toc_against_body(toc_headings, heading_map, heading_order)
    if not report.matches:
        raise ValueError("No table-of-contents headings could be validated against the document body")
    return pages, report


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

    pages, report = _load_pages_and_headings(markdown_path, toc_pages=toc_pages)
    return _segment_by_headings(pages, report.validated, delimiter=delimiter)


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

    pages, report = _load_pages_and_headings(args.markdown, toc_pages=args.toc_pages)
    headings = report.validated

    if args.print_toc:
        for match in report.matches:
            print(match.document_heading.text)
        _print_toc_report(report)
        return

    if report.has_issues:
        _print_toc_report(report)

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