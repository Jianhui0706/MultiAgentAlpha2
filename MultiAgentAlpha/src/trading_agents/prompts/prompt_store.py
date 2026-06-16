from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]


class PromptStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, key: str) -> PromptBundle:
        raw = self.load_mapping(key)
        return PromptBundle(
            system_prompt=_normalize_prompt_text(raw.get("system_prompt", "")),
            user_prompt=_normalize_prompt_text(raw.get("user_prompt", "")),
            metadata={k: v for k, v in raw.items() if k not in {"system_prompt", "user_prompt"}},
        )

    def load_mapping(self, key: str) -> dict[str, Any]:
        prompt_file = self._resolve_prompt_file(key)
        if prompt_file.suffix == ".md":
            return _load_markdown_prompt(prompt_file)
        if prompt_file.suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(prompt_file.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"invalid prompt YAML mapping: {prompt_file}")
            return dict(raw)
        raw = json.loads(prompt_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"invalid prompt JSON mapping: {prompt_file}")
        return dict(raw)

    def _resolve_prompt_file(self, key: str) -> Path:
        for suffix in (".md", ".yaml", ".yml", ".json"):
            prompt_file = self.root / f"{key}{suffix}"
            if prompt_file.exists():
                return prompt_file
        raise FileNotFoundError(f"prompt file not found for key {key!r} under {self.root}")


def _load_markdown_prompt(prompt_file: Path) -> dict[str, Any]:
    text = prompt_file.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    body = text
    if text.startswith("---\n") or text.startswith("---\r\n"):
        newline = "\r\n" if text.startswith("---\r\n") else "\n"
        marker = f"{newline}---{newline}"
        end = text.find(marker, len(f"---{newline}"))
        if end == -1:
            raise ValueError(f"missing closing front matter marker: {prompt_file}")
        front_matter = text[len(f"---{newline}") : end]
        loaded = yaml.safe_load(front_matter) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"invalid front matter mapping: {prompt_file}")
        metadata = dict(loaded)
        body = text[end + len(marker) :]

    sections = _split_markdown_prompt_sections(body, prompt_file)
    return {**metadata, **sections}


def _split_markdown_prompt_sections(body: str, prompt_file: Path) -> dict[str, str]:
    section_aliases = {
        "system prompt": "system_prompt",
        "user prompt": "user_prompt",
    }
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip().lower()
            current = section_aliases.get(title, _slugify_section_title(title))
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)

    missing = [name for name in ("system_prompt", "user_prompt") if not "\n".join(sections.get(name, [])).strip()]
    if missing:
        raise ValueError(f"missing markdown prompt section(s) {missing}: {prompt_file}")
    return {name: "\n".join(lines).strip() for name, lines in sections.items() if "\n".join(lines).strip()}


def _slugify_section_title(title: str) -> str:
    chars: list[str] = []
    previous_underscore = False
    for char in title:
        if char.isalnum():
            chars.append(char)
            previous_underscore = False
        elif not previous_underscore:
            chars.append("_")
            previous_underscore = True
    return "".join(chars).strip("_")


def _normalize_prompt_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def render_prompt(template: str, context: dict[str, object]) -> str:
    try:
        return Environment(undefined=StrictUndefined).from_string(template).render(**context)
    except Exception:
        rendered = template
        for key, value in context.items():
            text = "" if value is None else str(value)
            rendered = rendered.replace(f"{{{{ {key} }}}}", text)
            rendered = rendered.replace(f"{{{{{key}}}}}", text)
        return rendered
