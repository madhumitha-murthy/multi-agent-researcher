# platform/mcp_server.py - Model Context Protocol server
#
# Exposes research agents as MCP tools so any MCP-compatible client
# (Claude Desktop, VS Code, etc.) can invoke them directly.

import json
import os

from mcp.server.fastmcp import FastMCP

from research_platform.agent_registry import get_registry, register_default_agents
from research_platform.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("mcp_server")

mcp = FastMCP(
    name="multi-agent-researcher",
    instructions="Research platform with paper search, web search, fact checking, and report writing agents.",
)

# Ensure agents are registered
register_default_agents()


@mcp.tool()
def research(question: str) -> str:
    """Run a full multi-agent research pipeline on a question.
    Returns a structured report with citations and evidence."""
    from dotenv import load_dotenv
    load_dotenv()

    from graph import build_graph, run_research
    from llm import GeminiModel

    logger.info(f"MCP research invoked: {question[:80]}")
    model = GeminiModel()
    graph = build_graph(model)
    result = run_research(graph, question)
    return json.dumps({
        "report": result["report"],
        "supported_claims": result["supported_claims"],
        "total_claims": result["total_claims"],
        "stop_reason": result["stop_reason"],
        "latency_s": result["latency_s"],
    }, indent=2)


@mcp.tool()
def search_papers(query: str, max_results: int = 5) -> str:
    """Search academic papers on ArXiv and Semantic Scholar."""
    from tools.arxiv_search import search_arxiv
    from tools.s2_search import search_semantic_scholar

    arxiv_results = search_arxiv(query=query, max_results=max_results)
    s2_results = search_semantic_scholar(query=query, max_results=max_results)

    return json.dumps({
        "arxiv": arxiv_results,
        "semantic_scholar": s2_results,
    }, indent=2, default=str)


@mcp.tool()
def search_web_and_github(query: str, max_results: int = 5) -> str:
    """Search the web and GitHub for implementations and reports."""
    from tools.web_search import search_web
    from tools.github_search import search_github

    web_results = search_web(query=query, max_results=max_results)
    github_results = search_github(query=query, max_results=max_results)

    return json.dumps({
        "web": web_results,
        "github": github_results,
    }, indent=2, default=str)


@mcp.tool()
def list_agents() -> str:
    """List all registered research agents and their capabilities."""
    registry = get_registry()
    agents = registry.list_all()
    return json.dumps(
        [a.to_dict() for a in agents],
        indent=2,
    )


@mcp.tool()
def agent_health() -> str:
    """Check health status of all registered agents."""
    registry = get_registry()
    health = registry.check_health()
    return json.dumps({
        "health": health,
        "summary": registry.to_dict(),
    }, indent=2)


@mcp.resource("registry://agents")
def get_agent_registry() -> str:
    """Full agent registry as a resource."""
    registry = get_registry()
    return json.dumps(registry.to_dict(), indent=2)


def main():
    """Run the MCP server (stdio transport)."""
    logger.info("Starting MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
