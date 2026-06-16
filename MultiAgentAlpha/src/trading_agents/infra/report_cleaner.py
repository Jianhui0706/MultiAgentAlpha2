from __future__ import annotations

import re
from collections import Counter
from typing import Any


_DROP_LINE_PATTERNS = [
    r"请务必仔细阅读正文之后的各项信息披露与声明",
    r"证券研究报告",
    r"分析师",
    r"联系人",
    r"邮箱",
    r"电话",
    r"地址",
    r"网址",
    r"www\.",
    r"@\w+",
    r"页\s*\d+",
]

_DROP_BLOCK_PATTERNS = [
    r"免责声明",
    r"信息披露",
    r"法律声明",
    r"特别声明",
    r"风险提示",
    r"投资评级说明",
    r"分析师承诺",
    r"本公司具备证券投资咨询业务资格",
    r"申万宏源证券研究所",
    r"销售服务",
    r"客户服务",
    r"团队介绍",
    r"联系人",
    r"机构销售团队联系人",
    r"证券的投资评级",
    r"行业的投资评级",
    r"本报告采用的基准指数",
]

_KEEP_HINT_PATTERNS = [
    r"核心观点",
    r"结论",
    r"模型",
    r"策略",
    r"信号",
    r"因子",
    r"背离",
    r"动量",
    r"反转",
    r"波动",
    r"成交量",
    r"收益",
    r"预测",
    r"回测",
    r"实证",
    r"结果",
]


def _normalize_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return [line.strip() for line in normalized.split("\n")]


def _is_vertical_fragment_line(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False
    if compact.startswith("[Page "):
        return False
    if len(compact) > 3:
        return False
    if re.fullmatch(r"[A-Za-z0-9\W_]{1,3}", compact):
        return True
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", compact)
    return 0 < len(chinese_chars) <= 3 and len(compact) <= 3


def _is_repeated_noise_line(line: str, repeated_lines: set[str]) -> bool:
    compact = line.strip()
    if not compact:
        return False
    if compact in repeated_lines:
        return True
    return any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in _DROP_LINE_PATTERNS)


def _block_contains_drop_pattern(block: str) -> bool:
    return any(re.search(pattern, block, flags=re.IGNORECASE) for pattern in _DROP_BLOCK_PATTERNS)


def _block_contains_keep_hint(block: str) -> bool:
    return any(re.search(pattern, block, flags=re.IGNORECASE) for pattern in _KEEP_HINT_PATTERNS)


def clean_research_report_text(raw_text: str) -> dict[str, Any]:
    lines = _normalize_lines(raw_text)
    line_counter = Counter(line for line in lines if line and not line.startswith("[Page "))
    repeated_lines = {
        line
        for line, count in line_counter.items()
        if count >= 2 and len(line) <= 80
    }

    cleaned_lines: list[str] = []
    removed_snippets: list[str] = []
    removed_line_count = 0

    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue
        if _is_repeated_noise_line(line, repeated_lines):
            removed_line_count += 1
            if len(removed_snippets) < 20:
                removed_snippets.append(line)
            continue
        if _is_vertical_fragment_line(line):
            removed_line_count += 1
            if len(removed_snippets) < 20:
                removed_snippets.append(line)
            continue
        cleaned_lines.append(line)

    provisional_text = "\n".join(cleaned_lines)
    provisional_text = re.sub(r"\n{3,}", "\n\n", provisional_text).strip()

    cutoff_patterns = [
        r"\n3\.\s*风险提示与声明",
        r"\n风险提示与声明",
        r"\n信息披露",
        r"\n法律声明",
        r"\n机构销售团队联系人",
        r"\n证券的投资评级",
        r"\n行业的投资评级",
    ]
    cutoff_match_start: int | None = None
    for pattern in cutoff_patterns:
        match = re.search(pattern, provisional_text, flags=re.IGNORECASE)
        if match is None:
            continue
        start = match.start()
        if cutoff_match_start is None or start < cutoff_match_start:
            cutoff_match_start = start
    if cutoff_match_start is not None:
        tail_block = provisional_text[cutoff_match_start:].strip()
        if tail_block:
            if len(removed_snippets) < 20:
                removed_snippets.append(tail_block[:200])
            provisional_text = provisional_text[:cutoff_match_start].rstrip()

    blocks = [block.strip() for block in re.split(r"\n\s*\n", provisional_text) if block.strip()]
    kept_blocks: list[str] = []
    removed_block_count = 0

    for block in blocks:
        if _block_contains_drop_pattern(block) and not _block_contains_keep_hint(block):
            removed_block_count += 1
            if len(removed_snippets) < 20:
                removed_snippets.append(block[:200])
            continue
        kept_blocks.append(block)

    cleaned_text = "\n\n".join(kept_blocks).strip()
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return {
        "cleaned_text": cleaned_text,
        "removed_snippets": removed_snippets,
        "cleaning_stats": {
            "raw_char_count": len(raw_text),
            "cleaned_char_count": len(cleaned_text),
            "removed_line_count": removed_line_count,
            "removed_block_count": removed_block_count,
            "repeated_noise_line_count": len(repeated_lines),
        },
    }
