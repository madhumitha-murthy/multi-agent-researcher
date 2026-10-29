# agents/writer.py - Synthesises a structured research report from collected papers

SYSTEM_PROMPT = """You are a research report writer. Given a research question and a set of papers, write a clear, structured report.

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


def run_writer(model, question: str, papers: list[dict]) -> str:
    """Generate a structured research report."""
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

    messages = [{"role": "user", "text": prompt}]
    resp = model.call(SYSTEM_PROMPT, messages)
    return resp.text
