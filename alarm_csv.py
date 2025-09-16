"""Convert split sections into q/a/index CSV records."""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from split_sections import split_sections

# ``ALM-`` codes sometimes contain letters, numbers, underscores or dashes.
_CODE_RE = re.compile(r"ALM-[A-Za-z0-9_-]+")
_HEADER_RE = re.compile(
    r"^(?P<chapter>\d+(?:\.\d+)+)\s+(?P<code>ALM-[A-Za-z0-9_-]+)\s*(?P<title>.*)$"
)
_STEP_RE = re.compile(r"^步骤\s*\d+")


@dataclass
class Metadata:
    """Basic manual metadata used to build index entries."""

    vendor: str | None = None
    product: str | None = None
    doc: str | None = None
    version: str | None = None


@dataclass
class AlarmInfo:
    """Structured representation of an alarm section."""

    chapter: str | None
    code: str
    title: str
    desc: str
    impact: str
    causes: list[str]
    prereq: str
    steps: list[str]
    clear: str
    ref: str
    props: dict[str, str]
    page: int | None


def parse_metadata(markdown_path: Path) -> Metadata:
    """Return minimal metadata inferred from ``markdown_path``."""

    doc_name = markdown_path.stem
    return Metadata(doc=doc_name)


def contains_alarm_code(section: str) -> bool:
    """Return ``True`` if ``section`` contains an alarm identifier."""

    return bool(_CODE_RE.search(section))


def parse_alarm_info(section_text: str) -> AlarmInfo | None:
    """Extract structured alarm information from ``section_text``."""

    lines = section_text.splitlines()
    if not lines:
        return None

    heading = lines[0].strip()
    match = _HEADER_RE.match(heading)
    chapter = match.group("chapter") if match else None
    code = match.group("code") if match else _CODE_RE.search(heading or "")
    code = code if isinstance(code, str) else (code.group(0) if code else None)
    title = match.group("title") if match else heading

    if not code:
        return None

    blocks = _extract_blocks(lines[1:])

    desc = _collapse_paragraph(blocks.get("desc", []))
    impact = _collapse_paragraph(blocks.get("impact", []))
    causes = _parse_bullets(blocks.get("causes", []))
    prereq = _collapse_paragraph(blocks.get("prereq", []))
    steps = _parse_steps(blocks.get("steps", []))
    clear = _collapse_paragraph(blocks.get("clear", []))
    ref = _collapse_paragraph(blocks.get("ref", []))
    props = _parse_property_table(blocks.get("props", []))
    page = _locate_page(section_text)

    return AlarmInfo(
        chapter=chapter,
        code=code,
        title=title.strip() if title else code,
        desc=desc,
        impact=impact,
        causes=causes,
        prereq=prereq,
        steps=steps,
        clear=clear,
        ref=ref,
        props=props,
        page=page,
    )


def _extract_blocks(lines: Sequence[str]) -> dict[str, list[str]]:
    """Return a mapping from logical block names to their raw lines."""

    heading_map = {
        "告警解释": "desc",
        "对系统的影响": "impact",
        "可能原因": "causes",
        "前提条件": "prereq",
        "处理步骤": "steps",
        "清除方式": "clear",
        "参考": "ref",
        "告警属性": "props",
    }
    break_prefixes = ("告警参数", "附加信息", "定位信息")

    blocks: dict[str, list[str]] = {name: [] for name in heading_map.values()}
    current: str | None = None

    for raw_line in lines:
        line = raw_line.strip("\n")
        stripped = line.strip()
        if not stripped and current:
            blocks[current].append("")
            continue

        if stripped in heading_map:
            current = heading_map[stripped]
            continue

        if any(stripped.startswith(prefix) for prefix in break_prefixes):
            current = None
            continue

        if current:
            blocks[current].append(stripped)

    return blocks


def _collapse_paragraph(lines: Sequence[str]) -> str:
    """Collapse ``lines`` into a single paragraph while removing footers."""

    cleaned = [line.strip() for line in lines if _should_keep_line(line)]
    joined = " ".join(chunk for chunk in cleaned if chunk)
    return joined.strip()


def _parse_bullets(lines: Sequence[str]) -> list[str]:
    """Return bullet items extracted from ``lines``."""

    items: list[str] = []
    current: list[str] | None = None

    for raw in lines:
        if not _should_keep_line(raw):
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(("●", "-", "*", "•")):
            if current:
                items.append(" ".join(current).strip())
            token = stripped.lstrip("●-*•").strip()
            current = [token] if token else []
            continue
        if current is None:
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        items.append(" ".join(current).strip())

    cleaned: list[str] = []
    for item in items:
        candidate = item.strip()
        if candidate:
            cleaned.append(candidate)

    return cleaned


def _parse_steps(lines: Sequence[str]) -> list[str]:
    """Return ordered maintenance steps extracted from ``lines``."""

    steps: list[str] = []
    current: list[str] | None = None

    for raw in lines:
        if not _should_keep_line(raw):
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        if _STEP_RE.match(stripped):
            if current:
                steps.append(" ".join(current).strip())
            current = [stripped]
        else:
            if current is None:
                current = [stripped]
            else:
                current.append(stripped)

    if current:
        steps.append(" ".join(current).strip())

    return [step for step in steps if step]


def _parse_property_table(lines: Sequence[str]) -> dict[str, str]:
    """Parse the alarm property table block."""

    rows = [line.strip() for line in lines if _should_keep_line(line)]
    rows = [row for row in rows if row]
    if len(rows) < 2:
        return {}

    header_tokens = rows[0].split()
    value_tokens = rows[1].split()
    mapping = {key: value_tokens[i] for i, key in enumerate(header_tokens) if i < len(value_tokens)}

    props: dict[str, str] = {}
    severity = mapping.get("告警级别")
    autoclear = mapping.get("可自动清除")
    category = mapping.get("告警类型")

    if severity:
        props["severity"] = severity
    if autoclear:
        props["autoclear"] = autoclear
    if category:
        props["category"] = category

    return props


def _locate_page(section_text: str) -> int | None:
    """Best-effort detection of the page number for ``section_text``."""

    for line in reversed(section_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        match = re.search(r"(?:文档版本|版权所有).*?(\d{1,4})\s*$", stripped)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def _should_keep_line(line: str) -> bool:
    """Return ``True`` if ``line`` is actual content rather than a footer."""

    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("华为云 Stack"):
        return False
    if stripped.startswith("文档版本"):
        return False
    if "版权所有" in stripped:
        return False
    return True


def build_question(info: AlarmInfo) -> str:
    """Construct the combined question string."""

    title = info.title or info.code
    main = f"如何处理 {info.code}：{title}？"
    variants = [
        f"{title} 如何修复",
        f"{info.code} 处理步骤",
        f"{title} 故障排查方法",
    ]
    return " || ".join([main] + variants)


def build_answer(info: AlarmInfo) -> str:
    """Combine the structured answer sections into a single block."""

    parts = [
        f"告警解释：{_or_default(info.desc)}",
        f"影响：{_or_default(info.impact)}",
        "可能原因：" + format_list(info.causes),
        f"前提条件：{_or_default(info.prereq)}",
        "处理步骤：" + format_steps(info.steps),
        f"清除方式：{_or_default(info.clear)}",
        f"参考：{_or_default(info.ref)}",
    ]
    return "\n".join(parts)


def build_index(meta: Metadata, info: AlarmInfo, serial: int) -> str:
    """Build the index key/value payload."""

    vendor = meta.vendor or "unknown-vendor"
    product = meta.product or (meta.doc or "unknown-product")
    uid = f"{vendor}-{product}-{info.code}-{serial:04d}"
    kv_pairs = {
        "uid": uid,
        "vendor": meta.vendor,
        "product": meta.product,
        "doc": meta.doc,
        "ver": meta.version,
        "chap": info.chapter,
        "alm": info.code.split("-", 1)[-1],
        "page": info.page,
        "severity": info.props.get("severity"),
        "autoclear": info.props.get("autoclear"),
        "cat": info.props.get("category"),
    }
    return "; ".join(f"{key}={value}" for key, value in kv_pairs.items() if value)


def format_list(items: Sequence[str]) -> str:
    """Return a formatted list or ``"无"`` when empty."""

    filtered = [item for item in items if item]
    if not filtered:
        return "无"
    return "; ".join(filtered)


def format_steps(steps: Sequence[str]) -> str:
    """Format step entries for the answer payload."""

    filtered = [step for step in steps if step]
    if not filtered:
        return "无"
    return "\n".join(filtered)


def _or_default(value: str, default: str = "无") -> str:
    return value.strip() if value and value.strip() else default


def save_csv(records: Sequence[dict[str, str]], output_csv: Path) -> None:
    """Persist ``records`` as UTF-8 CSV with q/a/index headers."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["q", "a", "index"])
        writer.writeheader()
        writer.writerows(records)


def convert_markdown_to_csv(markdown_path: Path, output_csv: Path | None = None) -> list[dict[str, str]]:
    """Convert a markdown manual into CSV records and return them."""

    sections = split_sections(markdown_path)
    metadata = parse_metadata(markdown_path)
    records: list[dict[str, str]] = []

    serial = 1
    for section in sections:
        if not contains_alarm_code(section):
            continue
        info = parse_alarm_info(section)
        if not info:
            continue
        q = build_question(info)
        a = build_answer(info)
        index = build_index(metadata, info, serial)
        records.append({"q": q, "a": a, "index": index})
        serial += 1

    if output_csv is not None:
        save_csv(records, output_csv)

    return records


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Convert markdown alarm sections to CSV")
    parser.add_argument("markdown", type=Path, help="Path to the markdown manual")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination CSV file (default: <markdown stem>.csv)",
    )
    args = parser.parse_args(argv)

    target = args.output or args.markdown.with_suffix(".csv")
    records = convert_markdown_to_csv(args.markdown, target)
    print(f"Extracted {len(records)} alarm records from {args.markdown}")
    print(f"Saved CSV to {target}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
