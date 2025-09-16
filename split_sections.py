"""Split vendor manual markdown into per-chapter sections.

This module focuses on the ``split_sections`` helper requested in the design
notes.  The PDF is converted to markdown first (see :mod:`extract_text`), after
which the markdown is divided into the smallest numbered chapters.  Each
chapter is returned as a string and can optionally be exported into another
``.md`` file with ``=====`` separators for manual inspection.  Both the input
and exported files are normalized to use the ``.md`` extension so they can be
fed back into the debugging workflow directly.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from typing import Iterator, List, Sequence

__all__ = ["split_sections", "export_sections"]

# Matches headings that start with a numeric chapter index such as
# ``1.4.1`` or ``2.5.7.17``.  At least one ``.`` must be present so that
# simple numbered lists (``1.``, ``2.`` …) do not trigger a new section.
_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?P<label>\d+(?:\.\d+)+)\s+(?P<title>\S(?:.*\S)?)\s*$"
)

# Table-of-contents entries are littered with dotted leaders (``.....`` or
# ``····``).  Skipping them keeps the output focused on real section bodies.
_DOTTED_LEADER_RE = re.compile(r"[.·]{4,}")
_LONG_DOT_RUN_RE = re.compile(r"[.·]{6,}")
_CONTENT_MARKERS = ("告警解释", "处理步骤", "可能原因", "对系统的影响")


def _looks_like_toc_filler(line: str) -> bool:
    """Return ``True`` if ``line`` resembles a dotted table-of-contents leader."""

    candidate = line.strip()
    if not candidate:
        return False
    normalized = candidate.replace(" ", "")
    dot_count = normalized.count(".") + normalized.count("·")
    if dot_count < 3:
        return False
    allowed = {".", "·"} | set("0123456789")
    return all(char in allowed for char in normalized)

# Separator inserted between exported sections.
_DEFAULT_DELIMITER = "====="



def _ensure_markdown_suffix(path: Path) -> Path:
    """Return ``path`` with a ``.md`` suffix, replacing any existing suffix."""

    if path.suffix.lower() == ".md":
        return path
    return path.with_suffix(".md")


def _iter_markdown_lines(path: Path) -> Iterator[str]:
    """Yield cleaned lines from ``path``, skipping page markers."""

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("## Page"):
                # ``extract_text`` prefixes every page with ``## Page <n>``.
                # They are not part of the logical content.
                continue
            yield line


def _normalize_heading(line: str) -> str | None:
    """Return a cleaned heading string if ``line`` starts a chapter."""

    stripped = line.strip()
    if not stripped:
        return None
    if _DOTTED_LEADER_RE.search(stripped):
        return None
    match = _CHAPTER_HEADING_RE.match(stripped)
    if not match:
        return None
    label = match.group("label")
    title = re.sub(r"\s+", " ", match.group("title")).strip()
    return f"{label} {title}" if title else label


def _trim_trailing_blanks(lines: Sequence[str]) -> List[str]:
    """Remove blank lines from the end of ``lines`` while preserving order."""

    trimmed = list(lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return trimmed


def _normalize_section(lines: Sequence[str]) -> str:
    """Collapse insignificant whitespace inside a section."""

    trimmed = _trim_trailing_blanks(lines)
    if len(trimmed) > 1:
        # Drop blank lines directly following the heading.
        while len(trimmed) > 1 and not trimmed[1].strip():
            trimmed.pop(1)
    return "\n".join(trimmed).strip()


def export_sections(
    sections: Sequence[str],
    output_path: str | Path,
    *,
    delimiter: str = _DEFAULT_DELIMITER,
) -> None:
    """Write ``sections`` to ``output_path`` separated by ``delimiter`` lines.

    The destination is forced to use a ``.md`` suffix to keep the intermediate
    artifacts compatible with markdown-aware tooling.
    """

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
) -> list[str]:

    """Split the markdown manual into the smallest numbered chapters.

    Parameters
    ----------
    markdown_path:
        Path to a vendor manual exported as ``.md``.  A ``ValueError`` is
        raised when the suffix is not ``.md`` to avoid mixing in other text
        encodings accidentally.
    """


    path = Path(markdown_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() != ".md":
        raise ValueError(f"split_sections expects a .md file, got {path}")


    sections: list[str] = []
    current: list[str] = []
    toc_candidate = False
    content_started = False

    def finalize() -> None:
        nonlocal toc_candidate, content_started
        if not current:
            return
        body_lines = [line for line in current[1:] if line.strip()]
        if any(_LONG_DOT_RUN_RE.search(line) for line in body_lines[:4]):
            current.clear()
            toc_candidate = False
            return
        normalized = _normalize_section(current)
        if normalized:
            sections.append(normalized)
            if not content_started and any(marker in normalized for marker in _CONTENT_MARKERS):
                content_started = True
        current.clear()
        toc_candidate = False

    for line in _iter_markdown_lines(path):
        stripped = line.strip()

        if _LONG_DOT_RUN_RE.search(stripped) and not content_started:
            finalize()
            current.clear()
            toc_candidate = False
            continue

        if stripped == delimiter:
            finalize()
            continue

        heading = _normalize_heading(line)
        if heading:
            finalize()
            current.append(heading)
            toc_candidate = True
            continue

        if not current:
            # Ignore content that appears before the first numbered heading.
            continue

        if not stripped:
            current.append("")
            continue

        if toc_candidate and _looks_like_toc_filler(stripped):
            current.clear()
            toc_candidate = False
            continue

        current.append(stripped)
        toc_candidate = False

    finalize()

    return sections


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

    args = parser.parse_args(argv)

    sections = split_sections(args.markdown, delimiter=args.delimiter)
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
