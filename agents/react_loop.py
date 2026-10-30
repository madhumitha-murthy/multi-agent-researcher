# agents/react_loop.py - Shared ReAct loop for worker agents
#
# Used by paper_agent and web_agent. The loop:
# 1. Calls the model with available tools
# 2. If the model requests tool calls, executes them (respecting budget)
# 3. Returns messages, raw tool results, and stats

import time

import tools.registry as registry
from tools.url_reader import wrap_untrusted


def run_react(
    model,
    system_prompt: str,
    task: str,
    tool_schemas: list[dict],
    max_tool_calls: int,
    execute_fn=None,
    allowed_urls: set[str] | None = None,
) -> dict:
    """Run a ReAct loop for a worker agent.

    Args:
        model: ChatModel or FakeModel
        system_prompt: system prompt for the agent
        task: the task instruction
        tool_schemas: list of tool schema dicts to pass to the model
        max_tool_calls: hard limit on tool calls
        execute_fn: optional override for tool execution (for testing).
            Signature: (name, args) -> dict
        allowed_urls: set of URLs seen in tool results (for read_url restriction).
            If None, no restriction on read_url.

    Returns:
        dict with: messages, tool_results, seen_urls, stats (calls_used, tool_errors,
        input_tokens, output_tokens, latency_s)
    """
    if execute_fn is None:
        execute_fn = registry.execute_tool

    if allowed_urls is None:
        allowed_urls = set()

    messages = [{"role": "user", "text": f"Task: {task}"}]
    tool_results: list[dict] = []  # raw tool results with source text
    calls_used = 0
    tool_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    t0 = time.perf_counter()

    for _ in range(max_tool_calls + 2):  # extra iterations for the final answer
        tools = tool_schemas if calls_used < max_tool_calls else None
        resp = model.call(system_prompt, messages, tools)
        total_input_tokens += resp.input_tokens
        total_output_tokens += resp.output_tokens

        if resp.tool_calls and calls_used < max_tool_calls:
            messages.append({
                "role": "assistant", "text": resp.text,
                "tool_calls": resp.tool_calls, "raw": resp.raw,
            })

            for tc in resp.tool_calls:
                if calls_used >= max_tool_calls:
                    break

                name = tc["name"]
                args = tc.get("args", {})

                # For read_url, enforce allowed_urls restriction
                if name == "read_url" and allowed_urls is not None:
                    args = dict(args)
                    args["allowed_urls"] = allowed_urls

                result = execute_fn(name, args)
                calls_used += 1

                if "error" in result:
                    tool_errors += 1

                # Track URLs from results for read_url allowlist
                if "result" in result and isinstance(result["result"], list):
                    for item in result["result"]:
                        if item.get("url"):
                            allowed_urls.add(item["url"])

                tool_results.append({
                    "tool": name, "args": tc.get("args", {}), **result,
                })

                # Wrap result for the agent with untrusted tags.
                # ONLY send the wrapped text — never the raw result — so
                # prompt-injection content is always inside <untrusted_source> tags.
                wrapped = wrap_untrusted(result, args.get("url", ""))
                messages.append({
                    "role": "tool", "name": name,
                    "result": wrapped,
                })
        else:
            # Final answer
            break

    latency = round(time.perf_counter() - t0, 2)

    return {
        "messages": messages,
        "tool_results": tool_results,
        "seen_urls": allowed_urls,
        "stats": {
            "calls_used": calls_used,
            "tool_errors": tool_errors,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "latency_s": latency,
        },
    }
