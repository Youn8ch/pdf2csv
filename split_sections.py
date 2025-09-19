"""Split vendor manual markdown into sections using the table of contents."""
from __future__ import annotations

import argparse
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = ["split_sections", "export_sections", "_extract_toc_entries", "HeadingInfo"]

_DEFAULT_DELIMITER = "====="
_TOC_PAGE_LIMIT = 50

_DOTTED_LEADER_RE = re.compile(r"[.·]{2,}")
_ENTRY_SPLIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*)$")
_NUMBER_ONLY_RE = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass
class HeadingInfo:
    """Location metadata for a heading extracted from the document body."""

    text: str
    page_index: int
    line_index: int
    end_line_index: int

    def location_label(self) -> str:
        """Return a human-friendly ``page/line`` label for debugging."""

        return f"page {self.page_index + 1}, line {self.line_index + 1}"


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


def _normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace characters in ``text``."""

    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def _normalize_marker(text: str) -> str:
    """Collapse whitespace in ``text`` for easier marker comparisons."""

    return re.sub(r"\s+", "", text.replace("\u3000", ""))


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
    return normalized.lower() in {"目录", "目錄", "contents"}


def _clean_toc_text(text: str) -> str:
    """Normalise table-of-contents text by removing dotted leaders and page numbers."""

    candidate = text.replace("\u3000", " ")
    has_dots = bool(_DOTTED_LEADER_RE.search(candidate))
    candidate = _DOTTED_LEADER_RE.sub(" ", candidate)
    if has_dots:
        candidate = re.sub(r"\s+\d+\s*$", "", candidate)
    return _normalize_spaces(candidate)


def _extract_toc_entries(
    pages: Sequence[Sequence[str]], *, max_pages: int = _TOC_PAGE_LIMIT
) -> tuple[list[str], tuple[int, int] | None]:
    """Return headings discovered in the table of contents and the last TOC position."""

    entries: list[str] = []
    toc_started = False
    toc_entries_found = False
    current_label: str | None = None
    current_parts: list[str] = []
    last_position: tuple[int, int] | None = None

    def finalize() -> None:
        nonlocal current_label, current_parts, toc_entries_found
        if current_label and current_parts:
            title = _normalize_spaces(" ".join(current_parts))
            if title:
                entries.append(f"{current_label} {title}")
                toc_entries_found = True
        current_label = None
        current_parts = []

    for page_index, page in enumerate(pages):
        if page_index >= max_pages:
            break

        marker_in_page = any(_is_toc_marker(line.strip()) for line in page)
        page_has_dots = any(_DOTTED_LEADER_RE.search(line) for line in page)

        if not toc_started:
            if not marker_in_page:
                continue
            toc_started = True

        if toc_entries_found and not page_has_dots:
            finalize()
            break

        for line_index, raw_line in enumerate(page):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if not toc_started:
                if _is_toc_marker(stripped):
                    toc_started = True
                continue
            if _is_toc_marker(stripped):
                continue

            match = _ENTRY_SPLIT_RE.match(stripped)
            if match:
                finalize()
                label, rest = match.groups()
                current_label = label
                current_parts = []
                cleaned = _clean_toc_text(rest)
                if cleaned:
                    current_parts.append(cleaned)
                last_position = (page_index, line_index)
                continue

            if _NUMBER_ONLY_RE.match(stripped):
                finalize()
                current_label = stripped
                current_parts = []
                last_position = (page_index, line_index)
                continue

            if current_label is not None:
                cleaned = _clean_toc_text(stripped)
                if cleaned:
                    current_parts.append(cleaned)
                    last_position = (page_index, line_index)
                continue

        if toc_started and current_label is not None and not page_has_dots:
            finalize()
            break

    finalize()
    return entries, last_position

def _write_toc_debug(headings: Sequence[str], markdown_path: Path) -> Path:
    """Write extracted TOC headings to a sibling ``*_toc.txt`` file."""

    debug_path = markdown_path.with_name(markdown_path.stem + "_toc.txt")
    if headings:
        content = "\n".join(headings) + "\n"
    else:
        content = "No table-of-contents headings detected.\n"
    debug_path.write_text(content, encoding="utf-8")
    return debug_path


def _flatten_body_lines(
    pages: Sequence[Sequence[str]], start_after: tuple[int, int] | None
) -> tuple[list[str], dict[tuple[int, int], int]]:
    """Return body lines after the TOC and a lookup from location to line index."""

    body_lines: list[str] = []
    position_to_index: dict[tuple[int, int], int] = {}

    start_page = start_after[0] if start_after else 0
    start_line = start_after[1] + 1 if start_after else 0

    for page_index, page in enumerate(pages):
        if page_index < start_page:
            continue
        line_start = start_line if page_index == start_page else 0
        for line_index in range(line_start, len(page)):
            position_to_index[(page_index, line_index)] = len(body_lines)
            body_lines.append(page[line_index])
    return body_lines, position_to_index


def _next_nonempty_line(
    pages: Sequence[Sequence[str]], page_index: int, line_index: int
) -> tuple[int, int, str] | None:
    """Return the next non-empty line after ``page_index``/``line_index``."""

    for current_page in range(page_index, len(pages)):
        start = line_index + 1 if current_page == page_index else 0
        page = pages[current_page]
        for idx in range(start, len(page)):
            text = page[idx].strip()
            if text:
                return current_page, idx, text
        line_index = -1
    return None


def _previous_nonempty_line(
    pages: Sequence[Sequence[str]], page_index: int, line_index: int
) -> tuple[int, int, str] | None:
    """Return the previous non-empty line before ``page_index``/``line_index``."""

    for current_page in range(page_index, -1, -1):
        end = line_index if current_page == page_index else len(pages[current_page])
        page = pages[current_page]
        for idx in range(end - 1, -1, -1):
            text = page[idx].strip()
            if text:
                return current_page, idx, text
        line_index = len(page)
    return None


def _heading_context_score(pages: Sequence[Sequence[str]], info: HeadingInfo) -> int:
    """Return a heuristic score describing how ``info`` fits typical section context."""

    score = 0

    next_line = _next_nonempty_line(pages, info.page_index, info.end_line_index)
    if next_line is not None:
        _, _, text = next_line
        normalized = text.strip()
        lower = normalized.lower()
        if normalized.startswith("步骤") or lower.startswith("step"):
            score -= 2
        if normalized.startswith("参见") or normalized.startswith("详见") or lower.startswith("see"):
            score -= 2
        if normalized and normalized[0].isdigit():
            score -= 1
        if normalized and normalized[0] in {"●", "▪", "-", "–"}:
            score -= 1
        if any(keyword in normalized for keyword in ("告警", "简介", "概述", "说明", "介绍", "场景", "流程", "功能", "目的")):
            score += 1

    previous_line = _previous_nonempty_line(pages, info.page_index, info.line_index)
    if previous_line is not None:
        _, _, text = previous_line
        normalized = text.strip()
        lower = normalized.lower()
        if normalized.endswith("：") or lower.endswith(":"):
            score -= 1
        if "参见" in normalized or lower.startswith("see"):
            score -= 1

    return score


def _collect_heading_occurrences(
    pages: Sequence[Sequence[str]],
    start_after: tuple[int, int] | None,
    toc_headings: Sequence[str],
) -> list[tuple[str, HeadingInfo]]:
    """Collect document heading occurrences appearing after ``start_after``."""

    target_keys = {_normalize_heading_key(heading) for heading in toc_headings}
    if not target_keys:
        return []

    occurrences: list[tuple[str, HeadingInfo]] = []

    current_label: str | None = None
    current_parts: list[str] = []
    current_page: int | None = None
    current_line: int | None = None
    current_last_line: int | None = None

    def record_current_if_target() -> None:
        nonlocal current_label, current_parts, current_page, current_line, current_last_line
        if (
            current_label
            and current_parts
            and current_page is not None
            and current_line is not None
        ):
            title = _normalize_spaces(" ".join(current_parts))
            if title:
                heading = f"{current_label} {title}"
                key = _normalize_heading_key(heading)
                if key in target_keys:
                    end_line = current_last_line if current_last_line is not None else current_line
                    occurrences.append((key, HeadingInfo(heading, current_page, current_line, end_line)))

    def finalize() -> None:
        nonlocal current_label, current_parts, current_page, current_line, current_last_line
        record_current_if_target()
        current_label = None
        current_parts = []
        current_page = None
        current_line = None
        current_last_line = None

    start_page = start_after[0] if start_after else 0

    for page_index, page in enumerate(pages):
        if page_index < start_page:
            continue
        line_index = 0
        if start_after and page_index == start_page:
            line_index = start_after[1] + 1
        while line_index < len(page):
            raw_line = page[line_index]
            stripped = raw_line.strip()
            if not stripped:
                finalize()
                line_index += 1
                continue

            match = _ENTRY_SPLIT_RE.match(stripped)
            if match:
                finalize()
                label, rest = match.groups()
                current_label = label
                current_parts = []
                cleaned = _clean_toc_text(rest)
                if cleaned:
                    current_parts.append(cleaned)
                current_page = page_index
                current_line = line_index
                current_last_line = line_index
                record_current_if_target()
                line_index += 1
                continue

            if _NUMBER_ONLY_RE.match(stripped):
                finalize()
                current_label = stripped
                current_parts = []
                current_page = page_index
                current_line = line_index
                current_last_line = line_index
                record_current_if_target()
                line_index += 1
                continue

            if current_label is not None:
                cleaned = _clean_toc_text(stripped)
                if cleaned:
                    current_parts.append(cleaned)
                    current_last_line = line_index
                    record_current_if_target()
                line_index += 1
                continue

            line_index += 1

    finalize()
    return occurrences

def _split_body_by_headings(
    pages: Sequence[Sequence[str]],
    toc_headings: Sequence[str],
    start_after: tuple[int, int] | None,
) -> list[str]:
    """Split the document body into sections based on matched TOC headings."""

    if not toc_headings:
        return []

    body_lines, position_to_index = _flatten_body_lines(pages, start_after)
    occurrences = _collect_heading_occurrences(pages, start_after, toc_headings)
    if not occurrences:
        return []

    candidate_map: dict[str, deque[tuple[int, int, HeadingInfo]]] = {}
    for key, info in occurrences:
        global_index = position_to_index.get((info.page_index, info.line_index))
        if global_index is None:
            continue
        score = _heading_context_score(pages, info)
        candidate_map.setdefault(key, deque()).append((global_index, score, info))

    ordered_indices: list[tuple[int, HeadingInfo]] = []
    last_index = -1
    # Headings must be matched sequentially; once a TOC entry cannot be
    # resolved after the previous match we stop splitting to avoid jumping
    # ahead and producing misordered sections.
    for heading in toc_headings:
        key = _normalize_heading_key(heading)
        queue = candidate_map.get(key)
        if not queue:
            break

        while queue and queue[0][0] <= last_index:
            queue.popleft()
        if not queue:
            break

        best_pos: int | None = None
        best_index = -1
        best_score = float("-inf")
        chosen_info: HeadingInfo | None = None
        for idx, (global_index, score, info) in enumerate(queue):
            if global_index <= last_index:
                continue
            if (
                best_pos is None
                or score > best_score
                or (score == best_score and global_index < best_index)
            ):
                best_pos = idx
                best_index = global_index
                best_score = score
                chosen_info = info

        if best_pos is None or chosen_info is None:
            break

        for _ in range(best_pos):
            queue.popleft()
        start_idx, _score, _ = queue.popleft()
        last_index = start_idx
        ordered_indices.append((start_idx, chosen_info))

    sections: list[str] = []
    for idx, (start_idx, _info) in enumerate(ordered_indices):
        end_idx = ordered_indices[idx + 1][0] if idx + 1 < len(ordered_indices) else len(body_lines)
        lines = body_lines[start_idx:end_idx]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        section_text = "\n".join(lines).strip("\n")
        if section_text:
            sections.append(section_text)
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
    toc_headings, toc_last_position = _extract_toc_entries(pages, max_pages=toc_pages)
    if not toc_headings:
        raise ValueError("Failed to locate table-of-contents headings in the markdown file")
    if toc_last_position is None:
        raise ValueError("Failed to determine where the table of contents ends")

    _write_toc_debug(toc_headings, path)

    return _split_body_by_headings(pages, toc_headings, toc_last_position)


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

    pages = _read_pages(args.markdown)
    toc_headings, toc_last_position = _extract_toc_entries(pages, max_pages=args.toc_pages)
    if not toc_headings:
        parser.error("Failed to locate table-of-contents headings in the markdown file")
    if toc_last_position is None:
        parser.error("Failed to determine where the table of contents ends")

    debug_path = _write_toc_debug(toc_headings, args.markdown)

    if args.print_toc:
        for heading in toc_headings:
            print(heading)
        print(f"TOC headings written to {debug_path}")
        return

    sections = _split_body_by_headings(pages, toc_headings, toc_last_position)
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
