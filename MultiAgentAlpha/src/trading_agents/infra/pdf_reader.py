from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pymupdf() -> Any:
    try:
        import pymupdf as fitz  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            import fitz  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PDF extraction requires PyMuPDF. Install it with `uv sync` "
                "or add the `pymupdf` package to the active Python environment."
            ) from exc
    return fitz


def extract_pdf_text(pdf_path: str | Path) -> dict[str, Any]:
    path = Path(pdf_path).expanduser().resolve()
    fitz = _load_pymupdf()
    document = fitz.open(path)
    pages: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    try:
        for index, page in enumerate(document, start=1):
            page_text = _normalize_pdf_text(page.get_text("text"))
            pages.append(
                {
                    "page_number": index,
                    "char_count": len(page_text),
                    "text": page_text,
                }
            )
            if page_text:
                full_text_parts.append(f"[Page {index}]\n{page_text}")
    finally:
        document.close()
    full_text = "\n\n".join(part for part in full_text_parts if part).strip()
    return {
        "file_path": str(path),
        "file_name": path.name,
        "page_count": len(pages),
        "text": full_text,
        "pages": pages,
    }
