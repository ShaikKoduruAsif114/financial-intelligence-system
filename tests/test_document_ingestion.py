import rag
from ingest import build_document_chunks
from vision_analysis import summarize_chart_image


def test_build_document_chunks_preserve_structure():
    sources = [
        {
            "kind": "text",
            "content": "Revenue increased 12% in Q4.",
            "source": "annual_report.pdf",
            "page": 3,
            "section": "Executive Summary",
        },
        {
            "kind": "table",
            "content": "| Metric | Value |\n| --- | --- |\n| Revenue | 12.4M |",
            "source": "annual_report.pdf",
            "page": 9,
            "section": "Financial Results",
        },
        {
            "kind": "figure",
            "content": "Figure 1: Operating margin trend across FY2024.",
            "source": "annual_report.pdf",
            "page": 11,
            "section": "Chart Analysis",
        },
    ]

    chunks = build_document_chunks(sources, chunk_size=200, chunk_overlap=20)

    assert len(chunks) >= 3
    assert {chunk["kind"] for chunk in chunks} >= {"text", "table", "figure"}
    assert any("Revenue" in chunk["content"] for chunk in chunks if chunk["kind"] == "text")
    assert any("Metric" in chunk["content"] for chunk in chunks if chunk["kind"] == "table")


def test_summarize_chart_image_uses_model_output(monkeypatch):
    def fake_call(*args, **kwargs):
        return {"choices": [{"message": {"content": "Operating margin rose from 18% to 24% across FY2024."}}]}

    monkeypatch.setattr("vision_analysis.call_vision_model", fake_call)

    summary = summarize_chart_image("fake_chart.png")

    assert "Operating margin" in summary
    assert "18%" in summary or "24%" in summary


def test_format_context_for_prompt_includes_type_and_chart_evidence():
    context = [
        {
            "text": "Operating margin rose from 18% to 24% across FY2024.",
            "metadata": {"source": "quarterly_report.pdf", "page": 12, "kind": "figure", "section": "Chart analysis"},
        }
    ]

    formatted = rag.format_context_for_prompt(context)

    assert "TYPE: FIGURE" in formatted
    assert "quarterly_report.pdf" in formatted
    assert "Operating margin" in formatted
