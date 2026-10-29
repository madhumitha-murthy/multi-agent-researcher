# tools/github_search.py - Search GitHub repos (no auth needed for basic search)

import requests

TOOL_SCHEMA = {
    "name": "search_github",
    "description": (
        "Search GitHub for repositories related to a research topic. "
        "Useful for finding implementations, codebases, and open-source projects."
    ),
    "params": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for repos"},
            "max_results": {"type": "integer", "description": "Number of repos (1-10)"},
        },
        "required": ["query"],
    },
}


def search_github(query: str, max_results: int = 5) -> list[dict]:
    n = max(1, min(int(max_results), 10))
    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "per_page": n},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])

        return [
            {
                "title": r["full_name"],
                "abstract": (r.get("description") or "")[:500],
                "url": r["html_url"],
                "source": "github",
                "stars": r.get("stargazers_count", 0),
                "language": r.get("language", ""),
            }
            for r in items
        ]
    except Exception as e:
        return [{"error": f"GitHub search failed: {e}"}]
