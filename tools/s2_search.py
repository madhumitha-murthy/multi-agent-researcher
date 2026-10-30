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
            "paperId": p.get("paperId", ""),
        })
    return papers


# --- S2 Graph API: references and citations ---

S2_GRAPH_URL = "https://api.semanticscholar.org/graph/v1/paper"

GET_REFERENCES_SCHEMA = {
    "name": "get_references",
    "description": (
        "Get the reference list (papers cited BY a given paper). "
        "Use a Semantic Scholar paper ID or URL."
    ),
    "params": {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "Semantic Scholar paper ID or URL",
            },
            "max_results": {"type": "integer", "description": "Max references to return (1-20)"},
        },
        "required": ["paper_id"],
    },
}

GET_CITATIONS_SCHEMA = {
    "name": "get_citations",
    "description": (
        "Get papers that CITE a given paper. "
        "Use a Semantic Scholar paper ID or URL."
    ),
    "params": {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "Semantic Scholar paper ID or URL",
            },
            "max_results": {"type": "integer", "description": "Max citations to return (1-20)"},
        },
        "required": ["paper_id"],
    },
}


def _s2_graph_call(paper_id: str, endpoint: str, max_results: int = 10) -> list[dict]:
    """Shared helper for references/citations endpoints."""
    global _last_call
    n = max(1, min(int(max_results), 20))

    headers = {}
    api_key = os.getenv("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    fields = "title,abstract,url,year,citationCount,authors"
    url = f"{S2_GRAPH_URL}/{paper_id}/{endpoint}"
    params = {"fields": f"citedPaper.{fields}" if endpoint == "references" else f"citingPaper.{fields}",
              "limit": n}

    with _lock:
        elapsed = time.time() - _last_call
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 429:
                time.sleep(4)
                resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as e:
            return [{"error": f"S2 {endpoint} failed: {e}"}]
        finally:
            _last_call = time.time()

    key = "citedPaper" if endpoint == "references" else "citingPaper"
    papers = []
    for item in data:
        p = item.get(key, {})
        if not p or not p.get("title"):
            continue
        authors = p.get("authors", [])
        papers.append({
            "title": p["title"],
            "abstract": (p.get("abstract") or "")[:500],
            "url": p.get("url", ""),
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
            "source": "semantic_scholar",
            "authors": ", ".join(a.get("name", "") for a in (authors or [])[:3]),
            "paperId": p.get("paperId", ""),
        })
    return papers


def get_references(paper_id: str, max_results: int = 10) -> list[dict]:
    return _s2_graph_call(paper_id, "references", max_results)


def get_citations(paper_id: str, max_results: int = 10) -> list[dict]:
    return _s2_graph_call(paper_id, "citations", max_results)
