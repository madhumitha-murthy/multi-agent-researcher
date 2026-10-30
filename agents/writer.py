# agents/writer.py - Synthesises a structured research report from verified evidence
#
# v2: accepts Evidence objects. Every factual sentence must cite [E#] IDs.
# References section built in code (not by the LLM).
# Post-check: invalid citations stripped, citation_coverage computed.

import re

from evidence import Evidence

SYSTEM_PROMPT = """You are a research report writer. Given a research question and verified evidence, write a clear, structured report.

Format:
## Overview
Brief answer to the research question (2-3 sentences)

## Key Findings
- Finding 1 [E#][E#]
- Finding 2 [E#]
- ...

## Approaches Compared
Brief comparison of different approaches found

## Gaps & Open Questions
What's still missing or uncertain

## Limitations of This Report
What this report does not cover, and caveats about the evidence

Rules:
- Cite evidence IDs in brackets (e.g. [E1][E3]) for EVERY factual claim
- Be concise - each section should be 3-5 sentences max
- Only state what the evidence supports - don't make things up
- If evidence is insufficient, say so honestly
- Do NOT write a References section — it will be added automatically"""

MIN_EVIDENCE_FOR_REPORT = 3  # below this, writer flags needs_more_evidence


def run_writer(model, question: str, evidence) -> str | dict:
    """Generate a structured research report.

    Args:
        question: the research question
        evidence: list of Evidence objects (v2) or list of paper dicts (v1 compat)

    Returns:
        str: the report text (with references appended and post-check applied)
        dict: {"needs_more_evidence": "topic"} if evidence is too thin
    """
    # v1 compatibility
    if evidence and isinstance(evidence, list) and isinstance(evidence[0], dict):
        return _write_from_papers(model, question, evidence)

    if not evidence or len(evidence) < MIN_EVIDENCE_FOR_REPORT:
        topics = set(e.task for e in evidence) if evidence else set()
        return {
            "needs_more_evidence": f"Only {len(evidence) if evidence else 0} supported claims. "
                                   f"Topics covered: {', '.join(topics) if topics else 'none'}"
        }

    # Build evidence list for the prompt
    evidence_list = ""
    for e in evidence:
        evidence_list += (
            f"\n[{e.id}] Claim: {e.claim}"
            f"\n    Source: {e.source_title} ({e.source_type})"
            f"\n    Quote: \"{e.quote[:300]}\""
            f"\n    URL: {e.source_url}\n"
        )

    prompt = (
        f"Research question: {question}\n\n"
        f"Verified evidence ({len(evidence)} supported claims):\n{evidence_list}\n\n"
        f"Write the research report now. Cite evidence IDs like [E1][E3] for every factual claim.\n"
        f"Do NOT include a References section."
    )

    messages = [{"role": "user", "text": prompt}]
    resp = model.call(SYSTEM_PROMPT, messages)
    report = resp.text

    # Post-check: validate citations and build references
    valid_ids = {e.id for e in evidence}
    report, invalid_citations = _strip_invalid_citations(report, valid_ids)
    references = _build_references(report, evidence)
    coverage = _compute_citation_coverage(report)

    # Append code-built references
    if references:
        report = report.rstrip() + "\n\n" + references

    return report


def _strip_invalid_citations(report: str, valid_ids: set[str]) -> tuple[str, list[str]]:
    """Remove [E#] citations that don't exist or aren't supported. Return cleaned report + invalid list."""
    invalid = []

    def replace_fn(match):
        eid = match.group(1)
        if eid in valid_ids:
            return match.group(0)
        invalid.append(eid)
        return ""

    cleaned = re.sub(r"\[(E\d+)\]", replace_fn, report)
    # Clean up double spaces left by removals
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned, invalid


def _build_references(report: str, evidence: list[Evidence]) -> str:
    """Build a References section from evidence IDs actually cited in the report."""
    cited_ids = set(re.findall(r"\[(E\d+)\]", report))
    if not cited_ids:
        return ""

    # Build reference list in cited order
    id_to_evidence = {e.id: e for e in evidence}
    refs = []
    for eid in sorted(cited_ids, key=lambda x: int(x[1:])):
        e = id_to_evidence.get(eid)
        if e:
            refs.append(f"- [{eid}] {e.source_title} — {e.source_url}")

    if not refs:
        return ""
    return "## References\n" + "\n".join(refs)


def _compute_citation_coverage(report: str) -> float:
    """Fraction of non-heading, non-reference factual sentences that carry at least one [E#] citation."""
    # Strip references and headers
    ref_match = re.search(r"##\s*References", report, re.IGNORECASE)
    body = report[:ref_match.start()] if ref_match else report

    # Split into sentences (by . ! ? or newlines), then filter
    parts = re.split(r"(?<=[.!?])\s+|\n+", body)
    factual = []
    for part in parts:
        part = part.strip(" -*\t")
        if not part or part.startswith("#") or len(part) < 25:
            continue
        factual.append(part)

    if not factual:
        return 0.0

    cited = sum(1 for s in factual if re.search(r"\[E\d+\]", s))
    return round(cited / len(factual), 3)


# --- v1 compatibility ---

def _write_from_papers(model, question: str, papers: list[dict]) -> str:
    """v1 compatibility: write from paper dicts."""
    paper_list = ""
    for i, p in enumerate(papers, 1):
        authors = p.get("authors", "Unknown")
        year = p.get("year", "?")
        citations = p.get("citations", "")
        cite_str = f" | Citations: {citations}" if citations else ""
        paper_list += (
            f"\n[{i}] {p['title']}"
            f"\n    Authors: {authors} | Year: {year}{cite_str}"
            f"\n    URL: {p.get('url', 'N/A')}"
            f"\n    Abstract: {p.get('abstract', '')[:300]}\n"
        )

    prompt = (
        f"Research question: {question}\n\n"
        f"Papers to synthesise ({len(papers)} total):\n{paper_list}\n\n"
        f"Write the research report now."
    )

    old_system = """You are a research report writer. Given a research question and a set of papers, write a clear, structured report.

Format:
## Overview
Brief answer to the research question (2-3 sentences)

## Key Findings
- Finding 1 [Paper Title]
- Finding 2 [Paper Title]
- ...

## Approaches Compared
Brief comparison of different approaches found

## Gaps & Open Questions
What's still missing or uncertain

## References
Numbered list of papers with titles and URLs

Rules:
- Cite paper titles in brackets when making claims
- Be concise - each section should be 3-5 sentences max
- Only state what the papers support - don't make things up
- If papers are insufficient, say so honestly"""

    messages = [{"role": "user", "text": prompt}]
    resp = model.call(old_system, messages)
    return resp.text
