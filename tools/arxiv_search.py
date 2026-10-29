# tools/arxiv_search.py - Search ArXiv for research papers

import threading

import arxiv

from config import ARXIV_CATEGORIES, ARXIV_MAX_PER_QUERY

TOOL_SCHEMA = {
    "name": "search_arxiv",
    "description": "Search ArXiv for recent AI/ML/NLP research papers by topic.",
    "params": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Research topic to search for"},
            "max_results": {"type": "integer", "description": "Number of papers (1-10)"},
        },
        "required": ["query"],
    },
}

# Shared client and lock — ArXiv rate-limits aggressively if hit in parallel
_client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=3)
_lock = threading.Lock()


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    cat_filter = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    n = max(1, min(int(max_results), ARXIV_MAX_PER_QUERY))

    search = arxiv.Search(
        query=f"({query}) AND ({cat_filter})",
        max_results=n,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    with _lock:
        try:
            for r in _client.results(search):
                papers.append({
                    "title": r.title,
                    "abstract": r.summary[:500],
                    "url": r.entry_id,
                    "year": r.published.year,
                    "source": "arxiv",
                    "authors": ", ".join(a.name for a in r.authors[:3]),
                })
        except Exception as e:
            return [{"error": f"ArXiv search failed: {e}"}]
    return papers
