# tools/web_search.py - Web search using DuckDuckGo (no API key needed)

from duckduckgo_search import DDGS

TOOL_SCHEMA = {
    "name": "search_web",
    "description": (
        "Search the web for blog posts, news articles, reports, and other non-academic content. "
        "Use this for industry news, product launches, opinions, and topics not well covered by papers."
    ),
    "params": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Number of results (1-10)"},
        },
        "required": ["query"],
    },
}


def search_web(query: str, max_results: int = 5) -> list[dict]:
    n = max(1, min(int(max_results), 10))
    try:
        results = DDGS().text(query, max_results=n)
        return [
            {
                "title": r.get("title", ""),
                "abstract": r.get("body", "")[:500],
                "url": r.get("href", ""),
                "source": "web",
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": f"Web search failed: {e}"}]
