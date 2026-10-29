# tools/s2_search.py - Search Semantic Scholar for research papers

import os
import threading
import time

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

_lock = threading.Lock()
_last_call = 0.0


def search_semantic_scholar(query: str, max_results: int = 5, year: str = "") -> list[dict]:
    global _last_call
    n = max(1, min(int(max_results), S2_MAX_PER_QUERY))

    params = {"query": query, "limit": n, "fields": S2_FIELDS}
    if year:
        params["year"] = year

    headers = {}
    api_key = os.getenv("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    with _lock:
        # Enforce minimum 1.1s between S2 calls
        elapsed = time.time() - _last_call
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

        data = None
        for attempt in range(3):
            try:
                resp = requests.get(S2_API_URL, params=params, headers=headers, timeout=10)
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)  # 2s, 4s, 8s backoff
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json().get("data", [])
                break
            except requests.exceptions.HTTPError:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return [{"error": f"Semantic Scholar API failed: HTTP {resp.status_code}"}]
            except Exception as e:
                return [{"error": f"Semantic Scholar API failed: {e}"}]

        _last_call = time.time()

    if data is None:
        return [{"error": "Semantic Scholar API: exhausted retries (429)"}]

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
