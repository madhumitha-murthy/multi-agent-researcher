# agents/supervisor.py - Supervisor agent that coordinates worker agents
#
# One LLM call per turn, JSON mode. Decides which worker runs next.
# Never sees raw source text — only structured evidence summaries.

import json

from budget import Budget
from evidence import EvidenceStore
from llm import strip_json_fences

VALID_AGENTS = {"paper_agent", "web_agent", "fact_checker", "writer", "FINISH"}

SYSTEM_PROMPT = """You are a research supervisor coordinating a team of specialist agents.

Your agents:
- paper_agent: searches academic papers (ArXiv, Semantic Scholar). Use for scholarly evidence.
- web_agent: searches the web, GitHub repos, blog posts. Use for implementations, benchmarks, industry reports.
- fact_checker: verifies collected evidence claims against their source quotes. Call this before writing.
- writer: writes the final research report from verified evidence. Only call when you have enough supported claims.

Your job:
- Coordinate agents to answer the research question thoroughly
- You do NOT research or write yourself — you only decide which agent acts next
- Use paper_agent for academic evidence, web_agent for practical/implementation info
- Call fact_checker whenever there are unverified claims, especially before calling writer
- Call writer only when there are enough SUPPORTED claims covering the main angles
- After writer produces a report, return FINISH (unless the writer flagged missing evidence)
- Be aware of remaining budget; finish gracefully when budget is low

Output ONLY valid JSON:
{"next_agent": "paper_agent|web_agent|fact_checker|writer|FINISH", "task": "specific instruction", "reasoning": "brief explanation"}"""


def run_supervisor(
    model,
    question: str,
    evidence_store: EvidenceStore,
    worker_log: list[dict],
    budget: Budget,
) -> dict:
    """One supervisor turn. Returns routing decision.

    Returns:
        dict with: next_agent, task, reasoning, tokens_in, tokens_out
    """
    # Build context
    evidence_summary = evidence_store.summary_for_supervisor()
    n_supported = len(evidence_store.supported())
    n_unverified = len(evidence_store.unverified())

    # Show last 8 worker log entries
    recent_log = worker_log[-8:] if worker_log else []
    log_text = "\n".join(
        f"  [{e.get('agent', '?')}] {e.get('summary', '')}" for e in recent_log
    ) if recent_log else "  (none yet)"

    prompt = (
        f"Research question: {question}\n\n"
        f"Evidence status:\n{evidence_summary}\n\n"
        f"Supported claims: {n_supported} | Unverified: {n_unverified}\n\n"
        f"Recent worker activity:\n{log_text}\n\n"
        f"Budget: {budget.remaining_summary()}\n\n"
        f"What should happen next?"
    )

    messages = [{"role": "user", "text": prompt}]
    resp = model.call(SYSTEM_PROMPT, messages, json_mode=True)

    # Parse response
    try:
        parsed = json.loads(strip_json_fences(resp.text))
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")

        next_agent = parsed.get("next_agent", "").strip()
        task = parsed.get("task", "").strip()
        reasoning = parsed.get("reasoning", "").strip()

        if next_agent not in VALID_AGENTS:
            raise ValueError(f"Invalid agent: {next_agent}")

    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # Fallback: fact_checker if unverified, else writer if supported, else FINISH
        next_agent, task, reasoning = _fallback_routing(evidence_store)

    # Code-level guards
    next_agent, task, reasoning = _apply_guards(
        next_agent, task, reasoning, evidence_store, budget
    )

    return {
        "next_agent": next_agent,
        "task": task,
        "reasoning": reasoning,
        "tokens_in": resp.input_tokens,
        "tokens_out": resp.output_tokens,
    }


def _fallback_routing(store: EvidenceStore) -> tuple[str, str, str]:
    """Fallback when LLM output is unparseable."""
    if store.unverified():
        return "fact_checker", "verify all unverified claims", "LLM response unparseable, checking claims"
    if store.supported():
        return "writer", "write report from supported evidence", "LLM response unparseable, writing report"
    return "FINISH", "", "LLM response unparseable, no evidence to work with"


def _apply_guards(
    next_agent: str,
    task: str,
    reasoning: str,
    store: EvidenceStore,
    budget: Budget,
) -> tuple[str, str, str]:
    """Apply code-level guards that override LLM decisions."""
    n_supported = len(store.supported())
    n_unverified = len(store.unverified())

    # Cannot call writer with zero supported claims
    if next_agent == "writer" and n_supported == 0:
        if n_unverified > 0:
            return "fact_checker", "verify claims before writing", "guard: no supported claims, checking first"
        return "paper_agent", task or "find initial evidence", "guard: no evidence yet, searching first"

    # Budget exhausted: route to writer if possible, then FINISH
    budget_reason = budget.exhausted()
    if budget_reason and next_agent not in ("writer", "FINISH"):
        if n_supported > 0:
            return "writer", "write report with available evidence (budget exhausted)", f"guard: {budget_reason}"
        return "FINISH", "", f"guard: {budget_reason}, no supported evidence"

    return next_agent, task, reasoning
