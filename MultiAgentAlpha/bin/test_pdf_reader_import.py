from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_pdf_reader_uses_working_pymupdf_binding(tmp_path: Path):
    from trading_agents.infra.pdf_reader import extract_pdf_text
    import pymupdf

    pdf_path = tmp_path / "smoke.pdf"
    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "hello pdf")
        document.save(pdf_path)
    finally:
        document.close()

    payload = extract_pdf_text(pdf_path)

    assert payload["page_count"] == 1
    assert "hello pdf" in payload["text"]
