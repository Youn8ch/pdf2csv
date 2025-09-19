"""Tests for the table-of-contents splitting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from split_sections import _extract_toc_entries, split_sections


def test_extract_toc_entries_combines_split_numbering_lines() -> None:
    """Table-of-contents numbers split across lines are reconstructed."""

    pages = [
        [
            "目录",
            "1 概述 .......... 1",
            "1.1",
            " 告警监控是什么 .......... 2",
            "1.2",
            " 告警监控简介",
            " 发展历史 .......... 3",
        ]
    ]

    entries, last_position = _extract_toc_entries(pages)

    assert entries == [
        "1 概述",
        "1.1 告警监控是什么",
        "1.2 告警监控简介 发展历史",
    ]
    assert last_position == (0, 6)


def test_split_sections_generates_debug_toc(tmp_path: Path) -> None:
    """Splitting a document writes a TOC debug file and returns body sections."""

    markdown = tmp_path / "sample.md"
    markdown.write_text(
        "\n".join(
            [
                "## Page 1",
                "",
                "前言",
                "",
                "## Page 2",
                "",
                "目 录",
                "1 概述 .......... 1",
                "1.1",
                " 概述介绍 .......... 2",
                "",
                "## Page 3",
                "",
                "1",
                "概述",
                "正文 A",
                "",
                "## Page 4",
                "",
                "1.1",
                "概述介绍",
                "正文 B",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sections = split_sections(markdown)

    assert len(sections) == 2
    assert sections[0].splitlines()[:2] == ["1", "概述"]
    assert "正文 B" in sections[1]

    debug_path = markdown.with_name("sample_toc.txt")
    assert debug_path.exists()
    assert debug_path.read_text(encoding="utf-8").splitlines() == [
        "1 概述",
        "1.1 概述介绍",
    ]



def test_split_sections_avoids_cross_reference_headings(tmp_path: Path) -> None:
    """Text that references another heading should not trigger a new section."""

    markdown = tmp_path / "cross_ref.md"
    markdown.write_text(
        "\n".join(
            [
                "## Page 1",
                "",
                "目录",
                "1 第一节 .......... 1",
                "1.1 第二节 .......... 2",
                "",
                "## Page 2",
                "",
                "1 第一节",
                "第一节内容",
                "处理步骤：",
                "1.1 第二节",
                "第二节标题",
                "请参见下一节了解详情。",
                "仍在第一节的正文。",
                "",
                "## Page 3",
                "",
                "1.1 第二节",
                "第二节标题",
                "第二节的正文内容。",
            ]
        ),
        encoding="utf-8",
    )

    sections = split_sections(markdown)

    assert len(sections) == 2
    assert "请参见下一节" in sections[0]
    assert "第二节的正文内容" in sections[1]
    assert "请参见下一节" not in sections[1]

