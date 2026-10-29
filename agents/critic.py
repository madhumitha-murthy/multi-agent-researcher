# agents/critic.py - Evaluates collected papers, finds gaps, can request more research

import json

SYSTEM_PROMPT = """You are a research critic. You receive papers collected by research agents and evaluate them.

Your job:
1. Check if the papers adequately cover the research question
2. Remove duplicates or irrelevant papers
3. Identify any gaps in coverage
4. Decide: is more research needed, or do we have enough?

Rules:
- If there's a clear gap, specify what topic needs more research
- If coverage is adequate (5+ relevant papers covering different angles), approve
- Be specific about gaps, not vague

Output ONLY valid JSON:
{"verdict": "approve" or "more_research", "gaps": ["specific gap 1"], "keep_papers": [0, 1, 3], "reasoning": "why"}

keep_papers is a list of indices (0-based) of papers to keep."""


def run_critic(model, question: str, papers: list[dict], round_num: int) -> dict:
    """Evaluate papers and decide if more research is needed."""
    paper_list = ""
    for i, p in enumerate(papers):
        paper_list += (
            f"\n[{i}] {p['title']}"
            f"\n    Source: {p.get('source', '?')} | Year: {p.get('year', '?')}"
            f"\n    {p.get('abstract', '')[:200]}\n"
        )

    prompt = (
        f"Research question: {question}\n"
        f"Evaluation round: {round_num}\n"
        f"Papers collected ({len(papers)} total):\n{paper_list}\n"
        f"Evaluate these papers. Are there gaps? Should we search for more?"
    )

    messages = [{"role": "user", "text": prompt}]
    resp = model.call(SYSTEM_PROMPT, messages)

    try:
        parsed = json.loads(resp.text)
        verdict = parsed.get("verdict", "approve")
        gaps = parsed.get("gaps", [])
        keep_indices = parsed.get("keep_papers", list(range(len(papers))))
        reasoning = parsed.get("reasoning", "")

        # Filter papers to only kept ones
        kept = [papers[i] for i in keep_indices if i < len(papers)]

        return {
            "verdict": verdict,
            "gaps": gaps if verdict == "more_research" else [],
            "papers": kept if kept else papers,
            "reasoning": reasoning,
        }
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, approve with all papers
        return {
            "verdict": "approve",
            "gaps": [],
            "papers": papers,
            "reasoning": "Critic response parsing failed, approving as-is",
        }
