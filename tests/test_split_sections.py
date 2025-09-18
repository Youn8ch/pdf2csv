"""Tests for the table-of-contents splitting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from split_sections import _extract_toc_entries


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
