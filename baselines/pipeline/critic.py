# agents/critic.py - Evaluates collected papers, finds gaps, can request more research

import json

from llm import strip_json_fences

SYSTEM_PROMPT = """You are a research critic. You receive papers collected by research agents and evaluate them.

Your job:
1. Check if the papers adequately cover the research question
2. Remove duplicates or irrelevant papers (web pages, dictionaries, unrelated repos)
3. Identify any gaps in coverage
4. Decide: is more research needed, or do we have enough?

Rules:
- If there's a clear gap, specify what topic needs more research
- If coverage is adequate (5+ relevant papers covering different angles), approve
- Be specific about gaps, not vague
- Remove sources that are not actual research (dictionary definitions, restaurant pages, generic web content)

Output ONLY valid JSON:
{"verdict": "approve" or "more_research", "gaps": ["specific gap 1"], "keep_papers": [1, 2, 4], "reasoning": "why"}

keep_papers is a list of paper numbers (1-based, matching the numbering shown above) to keep."""


def run_critic(model, question: str, papers: list[dict], round_num: int) -> dict:
    """Evaluate papers and decide if more research is needed."""
    paper_list = ""
    for i, p in enumerate(papers, 1):
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
    resp = model.call(SYSTEM_PROMPT, messages, json_mode=True)

    try:
        parsed = json.loads(strip_json_fences(resp.text))
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")

        verdict = parsed.get("verdict", "approve")
        gaps = parsed.get("gaps", [])
        keep_indices = parsed.get("keep_papers", list(range(1, len(papers) + 1)))
        reasoning = parsed.get("reasoning", "")

        # Validate indices: 1-based from LLM, convert to 0-based
        kept = []
        for i in keep_indices:
            if isinstance(i, int) and 1 <= i <= len(papers):
                kept.append(papers[i - 1])

        return {
            "verdict": verdict,
            "gaps": gaps if verdict == "more_research" else [],
            "papers": kept if kept else papers,
            "reasoning": reasoning,
        }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return {
            "verdict": "approve",
            "gaps": [],
            "papers": papers,
            "reasoning": f"Critic response parsing failed, approving as-is. Raw: {resp.text[:300]}",
        }
