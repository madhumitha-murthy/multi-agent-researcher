# agents/collector.py - Searches for papers using tools (ArXiv, Semantic Scholar)
# This is the core agentic part - the LLM decides which tools to call

import json

from config import MAX_COLLECTOR_CALLS
import tools.registry as registry

SYSTEM_PROMPT = """You are a research collector. Your job is to find relevant information on a given topic.

You have 5 tools:
- search_arxiv: recent preprints in AI/ML/NLP
- search_semantic_scholar: published papers with citation counts
- search_web: blog posts, news articles, reports, industry content
- search_github: open-source repos and implementations
- read_url: fetch full text from a specific URL

Strategy:
- Use 2-3 different sources for breadth (e.g. papers + web + github)
- Use specific, targeted queries
- Use read_url only if you need more detail from a specific result
- Stop when you have 5-10 relevant results

Return a final text summary of what you found when done searching."""


def run_collector(model, sub_task: str, max_calls: int = MAX_COLLECTOR_CALLS) -> dict:
    """Run the collector agent loop for one sub-task. Returns papers found + trace."""
    messages = [{"role": "user", "text": f"Find research papers on: {sub_task}"}]
    papers = []
    trace = []
    calls_used = 0

    for _ in range(max_calls + 2):  # extra iterations for the final answer
        tools = registry.COLLECTOR_TOOLS if calls_used < max_calls else None
        resp = model.call(SYSTEM_PROMPT, messages, tools)

        # If model wants to call tools
        if resp.tool_calls and calls_used < max_calls:
            messages.append({
                "role": "assistant", "text": resp.text,
                "tool_calls": resp.tool_calls, "raw": resp.raw,
            })

            for tc in resp.tool_calls:
                result = registry.execute_tool(tc["name"], tc.get("args", {}))
                calls_used += 1
                trace.append({"tool": tc["name"], "args": tc["args"], **result})

                # Collect papers from successful results
                if "result" in result and isinstance(result["result"], list):
                    for p in result["result"]:
                        if p.get("title") and not p.get("error"):
                            papers.append(p)

                messages.append({"role": "tool", "name": tc["name"], "result": result})
        else:
            # Final answer - done searching
            trace.append({"action": "answer", "text": resp.text[:200]})
            break

    # Deduplicate by title
    seen = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return {
        "sub_task": sub_task,
        "papers": unique,
        "calls_used": calls_used,
        "trace": trace,
    }
