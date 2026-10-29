# tools/s2_search.py - Search Semantic Scholar for research papers

import requests

from config import S2_API_URL, S2_FIELDS, S2_MAX_PER_QUERY

TOOL_SCHEMA = {
    "name": "search_semantic_scholar",
    "description": "Search Semantic Scholar for research papers. Good for finding highly cited work.",
    "params": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Research topic to search for"},
            "max_results": {"type": "integer", "description": "Number of papers (1-10)"},
            "year": {"type": "string", "description": "Year filter, e.g. '2024-' for 2024 onwards"},
        },
        "required": ["query"],
    },
}


def search_semantic_scholar(query: str, max_results: int = 5, year: str = "") -> list[dict]:
    n = max(1, min(int(max_results), S2_MAX_PER_QUERY))

    params = {"query": query, "limit": n, "fields": S2_FIELDS}
    if year:
        params["year"] = year

    try:
        resp = requests.get(S2_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as e:
        return [{"error": f"Semantic Scholar API failed: {e}"}]

    papers = []
    for p in data:
        if not p.get("title"):
            continue
        authors = p.get("authors", [])
        papers.append({
            "title": p["title"],
            "abstract": (p.get("abstract") or "")[:500],
            "url": p.get("url", ""),
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
            "source": "semantic_scholar",
            "authors": ", ".join(a.get("name", "") for a in authors[:3]),
        })
    return papers
