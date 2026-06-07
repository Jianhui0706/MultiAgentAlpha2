from __future__ import annotations

import ast
import json
import re
from typing import Any


def _extract_json_object(text: str) -> str:
    """从文本中提取第一个完整的 JSON 对象（处理嵌套花括号）。"""
    start_idx = text.find('{')
    if start_idx == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]

    return text[start_idx:]


def _extract_fenced_json_block(text: str) -> str:
    """提取第一个 markdown fenced code block 中的内容。"""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _strip_thinking_blocks(text: str) -> str:
    """移除思考块（如 {{-- ... --}} 或  Theresa...）。"""
    text = re.sub(r'\{\{--.*?--\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<think\s*>.*?</think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    """移除对象和数组中的尾随逗号。"""
    # 移除对象中的尾随逗号: {a: 1, b: 2,} -> {a: 1, b: 2}
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _fix_invalid_escapes(text: str) -> str:
    r"""修复无效的转义序列（如 LaTeX 中的 \_）。"""
    # 移除无效的转义序列
    text = re.sub(r'\\_', '_', text)
    text = re.sub(r'\\,', ',', text)
    text = re.sub(r'\\{', '{', text)
    text = re.sub(r'\\}', '}', text)
    return text


def _fix_unquoted_keys(text: str) -> str:
    """尝试修复未加引号的 JSON 键。"""
    # 简单的未加引号键修复，可能无法处理所有情况
    # 匹配 { key: 或 , key: 的模式
    text = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    return text


def _remove_control_characters(text: str) -> str:
    """移除 JSON 不接受的控制字符。"""
    return "".join(ch for ch in text if ch in ("\n", "\r", "\t") or ord(ch) >= 32)


def _fix_mismatched_closing_delimiters(text: str) -> str:
    """Repair common LLM typo where an array/object is closed with the wrong bracket."""
    result: list[str] = []
    stack: list[str] = []
    in_string = False
    escape_next = False
    expected_close = {"{": "}", "[": "]"}

    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            result.append(ch)
            continue
        if ch in "{[":
            stack.append(ch)
            result.append(ch)
            continue
        if ch in "}]":
            if stack:
                opener = stack.pop()
                result.append(expected_close[opener])
            else:
                result.append(ch)
            continue
        result.append(ch)

    while stack:
        result.append(expected_close[stack.pop()])
    return "".join(result)


def _escape_newlines_in_strings(text: str) -> str:
    """Escape bare newlines/tabs inside JSON string values so they don't break parsing."""
    result: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == "\n":
            result.append("\\n")
            continue
        if in_string and ch == "\r":
            result.append("\\r")
            continue
        if in_string and ch == "\t":
            result.append("\\t")
            continue
        result.append(ch)
    return "".join(result)


def _repair_unescaped_quotes(text: str) -> str:
    """修复字符串值中未转义的引号。"""
    result: list[str] = []
    stack: list[str] = []
    in_string = False
    escape_next = False
    string_role: str | None = None
    expecting_key = False
    expecting_value = False

    def next_nonspace(index: int) -> str:
        j = index + 1
        while j < len(text) and text[j].isspace():
            j += 1
        return text[j] if j < len(text) else ""

    for i, ch in enumerate(text):
        if in_string:
            if escape_next:
                result.append(ch)
                escape_next = False
                continue
            if ch == "\\":
                result.append(ch)
                escape_next = True
                continue
            if ch == '"':
                nxt = next_nonspace(i)
                if string_role == "key":
                    if nxt == ":":
                        in_string = False
                        string_role = None
                        result.append(ch)
                        continue
                    result.append('\\"')
                    continue
                if string_role == "value":
                    if nxt in {",", "}", "]", ""}:
                        in_string = False
                        string_role = None
                        result.append(ch)
                        continue
                    result.append('\\"')
                    continue
                in_string = False
                result.append(ch)
                continue
            result.append(ch)
            continue

        if ch == "{":
            stack.append("{")
            expecting_key = True
            expecting_value = False
            result.append(ch)
            continue
        if ch == "[":
            stack.append("[")
            expecting_key = False
            expecting_value = True
            result.append(ch)
            continue
        if ch == "}":
            if stack:
                stack.pop()
            expecting_key = bool(stack and stack[-1] == "{")
            expecting_value = False
            result.append(ch)
            continue
        if ch == "]":
            if stack:
                stack.pop()
            expecting_key = False
            expecting_value = False
            result.append(ch)
            continue
        if ch == ":":
            expecting_key = False
            expecting_value = True
            result.append(ch)
            continue
        if ch == ",":
            expecting_key = bool(stack and stack[-1] == "{")
            expecting_value = bool(stack and stack[-1] == "[")
            result.append(ch)
            continue
        if ch == '"':
            in_string = True
            string_role = "key" if expecting_key else "value" if expecting_value else None
            expecting_value = False
            result.append(ch)
            continue
        if not ch.isspace() and expecting_value:
            expecting_value = False
        result.append(ch)

    return "".join(result)


def parse_json_object(text: str) -> dict[str, Any]:
    """从可能包含非 JSON 文本的字符串中提取 JSON 对象。"""
    stripped = text.strip()

    stripped = _strip_thinking_blocks(stripped)
    stripped = _extract_fenced_json_block(stripped)
    stripped = _remove_control_characters(stripped)

    # 如果没有以 { 开头，尝试提取 JSON 对象
    if not stripped.startswith("{"):
        stripped = _extract_json_object(stripped)

    # 尝试 1: 原始解析
    try:
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("parsed json is not an object")
        return parsed
    except json.JSONDecodeError:
        pass

    # 尝试 2: 转义字符串值内的裸换行（LLM 多行文本最常见错误）
    fixed = _escape_newlines_in_strings(stripped)
    try:
        parsed = json.loads(fixed)
        if not isinstance(parsed, dict):
            raise ValueError("parsed json is not an object")
        return parsed
    except json.JSONDecodeError:
        pass

    # 尝试 3: 修复尾随逗号（最常见的错误）
    fixed = _fix_trailing_commas(fixed)
    try:
        parsed = json.loads(fixed)
        if not isinstance(parsed, dict):
            raise ValueError("parsed json is not an object")
        return parsed
    except json.JSONDecodeError:
        pass

    # 尝试 4: 修复无效转义
    fixed = _fix_invalid_escapes(fixed)
    try:
        parsed = json.loads(fixed)
        if not isinstance(parsed, dict):
            raise ValueError("parsed json is not an object")
        return parsed
    except json.JSONDecodeError:
        pass

    # 尝试 5: 修复对象/数组闭合符错配
    fixed = _fix_mismatched_closing_delimiters(fixed)
    try:
        parsed = json.loads(fixed)
        if not isinstance(parsed, dict):
            raise ValueError("parsed json is not an object")
        return parsed
    except json.JSONDecodeError:
        pass

    # 尝试 5: 修复未加引号的键
    fixed = _fix_unquoted_keys(fixed)
    try:
        parsed = json.loads(fixed)
        if not isinstance(parsed, dict):
            raise ValueError("parsed json is not an object")
        return parsed
    except json.JSONDecodeError:
        pass

    # 尝试 6: 修复字符串内未转义引号
    fixed = _repair_unescaped_quotes(fixed)
    try:
        parsed = json.loads(fixed)
        if not isinstance(parsed, dict):
            raise ValueError("parsed json is not an object")
        return parsed
    except json.JSONDecodeError:
        pass

    # 尝试 6b: 修复引号后，字符串边界已正确，再次转义裸换行
    fixed = _escape_newlines_in_strings(fixed)
    try:
        parsed = json.loads(fixed)
        if not isinstance(parsed, dict):
            raise ValueError("parsed json is not an object")
        return parsed
    except json.JSONDecodeError:
        pass

    # 尝试 7: Python 字面量兜底
    python_like = fixed.replace("true", "True").replace("false", "False").replace("null", "None")
    try:
        parsed = ast.literal_eval(python_like)
        if not isinstance(parsed, dict):
            raise ValueError("parsed json is not an object")
        return parsed
    except Exception as exc:
        raise ValueError(f"JSON parse failed after all attempts. Original error: {exc}")
