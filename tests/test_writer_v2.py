# tests/test_writer_v2.py - Tests for the v2 writer with evidence-ID citations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.writer import (
    _build_references,
    _compute_citation_coverage,
    _strip_invalid_citations,
    run_writer,
)
from evidence import Evidence
from llm import FakeModel, ModelResponse


def _make_evidence(n=4):
    """Create n Evidence objects for testing."""
    items = []
    for i in range(1, n + 1):
        items.append(Evidence(
            id=f"E{i}",
            claim=f"Claim {i} about the topic",
            source_title=f"Paper {i}",
            source_url=f"http://paper{i}",
            source_type="arxiv",
            quote=f"Quote from paper {i}",
            found_by="paper_agent",
            task="test task",
            status="supported",
        ))
    return items


# --- Citation stripping ---

def test_strip_invalid_citations_keeps_valid():
    text = "This is a claim [E1] and another [E3]."
    valid = {"E1", "E2", "E3"}
    result, invalid = _strip_invalid_citations(text, valid)
    assert "[E1]" in result
    assert "[E3]" in result
    assert len(invalid) == 0


def test_strip_invalid_citations_removes_invalid():
    text = "This claim [E1] cites [E99] which doesn't exist."
    valid = {"E1", "E2"}
    result, invalid = _strip_invalid_citations(text, valid)
    assert "[E1]" in result
    assert "[E99]" not in result
    assert "E99" in invalid


def test_strip_invalid_citations_all_invalid():
    text = "Claims [E50][E51] are both invalid."
    valid = {"E1"}
    result, invalid = _strip_invalid_citations(text, valid)
    assert "[E50]" not in result
    assert "[E51]" not in result
    assert len(invalid) == 2


# --- Code-built references ---

def test_build_references_from_cited_ids():
    evidence = _make_evidence(3)
    report = "Finding A [E1]. Finding B [E3]."
    refs = _build_references(report, evidence)
    assert "## References" in refs
    assert "[E1]" in refs
    assert "[E3]" in refs
    assert "Paper 1" in refs
    assert "Paper 3" in refs
    # E2 not cited, should not appear
    assert "Paper 2" not in refs


def test_build_references_empty_when_no_citations():
    evidence = _make_evidence(3)
    report = "A report with no citations at all."
    refs = _build_references(report, evidence)
    assert refs == ""


# --- Citation coverage ---

def test_citation_coverage_full():
    report = "## Overview\nThis topic is important [E1]. It has many aspects [E2]."
    coverage = _compute_citation_coverage(report)
    assert coverage == 1.0


def test_citation_coverage_partial():
    report = "## Overview\nThis topic is important [E1]. This line has no citation and is long enough."
    coverage = _compute_citation_coverage(report)
    assert coverage == 0.5


def test_citation_coverage_none():
    report = "## Overview\nThis is a long line without any evidence citations at all."
    coverage = _compute_citation_coverage(report)
    assert coverage == 0.0


# --- Full writer flow ---

def test_writer_produces_report_with_evidence():
    evidence = _make_evidence(4)
    model = FakeModel([
        ModelResponse(text=(
            "## Overview\nThis report covers the topic thoroughly [E1][E2].\n\n"
            "## Key Findings\n- Finding A based on research [E1]\n- Finding B from experiments [E3]\n\n"
            "## Limitations of This Report\nLimited to 4 sources."
        ))
    ])
    result = run_writer(model, "test question", evidence)
    assert isinstance(result, dict)
    report = result["report"]
    assert "## Overview" in report
    assert "## References" in report
    assert "[E1]" in report
    assert result["citation_coverage"] > 0
    assert result["input_tokens"] > 0


def test_writer_strips_invalid_citations_in_report():
    evidence = _make_evidence(3)
    model = FakeModel([
        ModelResponse(text="## Overview\nClaim [E1] and fabricated [E99] and real [E2].")
    ])
    result = run_writer(model, "test question", evidence)
    report = result["report"]
    assert "[E1]" in report
    assert "[E2]" in report
    assert "[E99]" not in report  # stripped
    assert "E99" in result["invalid_citations"]


def test_writer_returns_needs_more_evidence_when_insufficient():
    # Only 2 evidence items (below MIN_EVIDENCE_FOR_REPORT = 3)
    evidence = _make_evidence(2)
    model = FakeModel([])  # should not be called
    result = run_writer(model, "test question", evidence)
    assert isinstance(result, dict)
    assert "needs_more_evidence" in result


def test_writer_returns_needs_more_evidence_when_empty():
    model = FakeModel([])
    result = run_writer(model, "test question", [])
    assert isinstance(result, dict)
    assert "needs_more_evidence" in result


def test_writer_references_only_contain_cited_evidence():
    """References section should only contain evidence IDs that appear in the report body."""
    evidence = _make_evidence(5)
    model = FakeModel([
        ModelResponse(text="## Overview\nOnly citing E1 and E3 [E1][E3].\n\n## Key Findings\n- One finding [E1]")
    ])
    result = run_writer(model, "test question", evidence)
    report = result["report"]
    # References should have E1 and E3, but NOT E2, E4, E5
    assert "Paper 1" in report
    assert "Paper 3" in report
    # E2, E4, E5 should not be in references
    ref_section = report[report.index("## References"):]
    assert "Paper 2" not in ref_section
    assert "Paper 4" not in ref_section
    assert "Paper 5" not in ref_section
