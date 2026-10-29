# agents/planner.py - Decomposes a research question into sub-tasks

import json

SYSTEM_PROMPT = """You are a research planner. Given a research question, break it into 2-3 focused sub-tasks that can be searched independently.

Rules:
- Each sub-task should be a specific search query
- Cover different angles of the question
- Return ONLY valid JSON, no other text

Output format:
{"sub_tasks": ["search query 1", "search query 2", "search query 3"]}"""


def run_planner(model, question: str) -> list[str]:
    """Use the LLM to decompose a question into sub-tasks."""
    messages = [{"role": "user", "text": question}]
    resp = model.call(SYSTEM_PROMPT, messages)

    try:
        parsed = json.loads(resp.text)
        tasks = parsed.get("sub_tasks", [])
        if isinstance(tasks, list) and len(tasks) >= 1:
            return tasks[:3]
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: if LLM didn't return valid JSON, use the original question
    return [question]
