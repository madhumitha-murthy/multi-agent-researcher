# tools/registry.py - Central tool registry

from tools.arxiv_search import TOOL_SCHEMA as ARXIV_SCHEMA, search_arxiv
from tools.github_search import TOOL_SCHEMA as GITHUB_SCHEMA, search_github
from tools.s2_search import TOOL_SCHEMA as S2_SCHEMA, search_semantic_scholar
from tools.url_reader import TOOL_SCHEMA as URL_SCHEMA, read_url
from tools.web_search import TOOL_SCHEMA as WEB_SCHEMA, search_web

COLLECTOR_TOOLS = [ARXIV_SCHEMA, S2_SCHEMA, WEB_SCHEMA, GITHUB_SCHEMA, URL_SCHEMA]

TOOL_FUNCTIONS = {
    "search_arxiv": search_arxiv,
    "search_semantic_scholar": search_semantic_scholar,
    "search_web": search_web,
    "search_github": search_github,
    "read_url": read_url,
}


def execute_tool(name: str, args: dict) -> dict:
    """Run a tool by name. Returns result or error dict."""
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        result = fn(**args)
        # read_url returns a single dict, not a list
        if isinstance(result, dict) and "error" not in result:
            return {"result": [result] if name == "read_url" else result}
        if isinstance(result, dict) and "error" in result:
            return result
        return {"result": result}
    except Exception as e:
        return {"error": f"{name} failed: {e}"}
