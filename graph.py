# graph.py - Supervisor-routed LangGraph multi-agent orchestrator (v2)
#
# Pattern: hub-and-spoke. Workers always return to the supervisor.
# The supervisor never sees raw source text, only structured evidence summaries.
# Human-in-the-loop: graph pauses before publishing a report for admin approval.

import time
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.fact_checker import run_fact_checker
from agents.paper_agent import run_paper_agent
from agents.supervisor import run_supervisor
from agents.web_agent import run_web_agent
from agents.writer import run_writer
from budget import Budget
from config import MAX_SUPERVISOR_TURNS, MAX_WORKER_TOOL_CALLS
from evidence import EvidenceStore


class ResearchState(TypedDict):
    question: str
    evidence: dict              # EvidenceStore.to_dict()
    next_agent: str             # set by supervisor
    next_task: str              # instruction for the chosen worker
    supervisor_turns: int
    worker_log: list[dict]      # short result summary per worker call
    report: str
    budget: dict                # Budget.to_dict()
    trace: list[dict]
    stop_reason: str            # "finished" | "turn_limit" | "cost_limit" | "error"
    source_texts: dict          # url -> text, accumulated from workers
    approval_status: str        # "" | "pending" | "approved" | "rejected"
    approval_feedback: str      # reviewer feedback (on reject)


def build_graph(model, execute_fn=None, checkpointer=None):
    """Build the supervisor-routed multi-agent graph.

    Args:
        model: ChatModel or FakeModel
        execute_fn: optional tool execution override (for testing)
        checkpointer: LangGraph checkpointer for state persistence (default: MemorySaver)
    """

    def supervisor_node(state: ResearchState) -> dict:
        budget = Budget.from_dict(state["budget"])
        store = EvidenceStore.from_dict(state["evidence"])

        # Check budget before calling LLM
        budget_reason = budget.exhausted()
        if budget_reason:
            n_supported = len(store.supported())
            if n_supported > 0 and not state.get("report"):
                return {
                    "next_agent": "writer",
                    "next_task": "write report with available evidence (budget exhausted)",
                    "supervisor_turns": state["supervisor_turns"] + 1,
                    "trace": state["trace"] + [{
                        "agent": "supervisor", "decision": "writer",
                        "reason": f"budget exhausted: {budget_reason}",
                    }],
                    "budget": budget.to_dict(),
                }
            return {
                "next_agent": "FINISH",
                "next_task": "",
                "stop_reason": budget_reason,
                "supervisor_turns": state["supervisor_turns"] + 1,
                "trace": state["trace"] + [{
                    "agent": "supervisor", "decision": "FINISH",
                    "reason": f"budget exhausted: {budget_reason}",
                }],
                "budget": budget.to_dict(),
            }

        # Guard: once a report exists, FINISH — unless the writer asked for more evidence
        if state.get("report"):
            last_writer = None
            for entry in reversed(state.get("worker_log", [])):
                if entry.get("agent") == "writer":
                    last_writer = entry
                    break
            if not (last_writer and last_writer.get("needs_more_evidence")):
                return {
                    "next_agent": "FINISH",
                    "next_task": "",
                    "stop_reason": "finished",
                    "supervisor_turns": state["supervisor_turns"] + 1,
                    "trace": state["trace"] + [{
                        "agent": "supervisor", "decision": "FINISH",
                        "reason": "report already written",
                    }],
                    "budget": budget.to_dict(),
                }

        t0 = time.perf_counter()
        result = run_supervisor(
            model=model,
            question=state["question"],
            evidence_store=store,
            worker_log=state["worker_log"],
            budget=budget,
        )
        latency = round(time.perf_counter() - t0, 2)

        budget.record_turn()
        budget.record_tokens(result["tokens_in"], result["tokens_out"])

        return {
            "next_agent": result["next_agent"],
            "next_task": result["task"],
            "supervisor_turns": state["supervisor_turns"] + 1,
            "budget": budget.to_dict(),
            "trace": state["trace"] + [{
                "agent": "supervisor",
                "decision": result["next_agent"],
                "task": result["task"],
                "reason": result["reasoning"],
                "latency_s": latency,
            }],
        }

    def paper_agent_node(state: ResearchState) -> dict:
        store = EvidenceStore.from_dict(state["evidence"])
        budget = Budget.from_dict(state["budget"])
        source_texts = dict(state.get("source_texts", {}))

        try:
            t0 = time.perf_counter()
            result = run_paper_agent(
                model=model,
                task=state["next_task"],
                evidence_store=store,
                max_tool_calls=MAX_WORKER_TOOL_CALLS,
                execute_fn=execute_fn,
            )
            latency = round(time.perf_counter() - t0, 2)

            budget.record_tool_calls(result["tool_calls"])
            budget.record_tokens(result["tokens_in"], result["tokens_out"])
            source_texts.update(result.get("source_texts", {}))

            log_entry = {
                "agent": "paper_agent",
                "task": state["next_task"],
                "evidence_added": result["evidence_added"],
                "summary": result["summary"],
                "tool_calls": result["tool_calls"],
                "tool_errors": result["tool_errors"],
                "tokens_in": result["tokens_in"],
                "tokens_out": result["tokens_out"],
                "latency_s": latency,
            }
        except Exception as e:
            log_entry = {
                "agent": "paper_agent",
                "task": state["next_task"],
                "evidence_added": [],
                "summary": f"Error: {str(e)[:200]}",
                "error": str(e)[:200],
            }

        return {
            "evidence": store.to_dict(),
            "worker_log": state["worker_log"] + [log_entry],
            "budget": budget.to_dict(),
            "source_texts": source_texts,
            "trace": state["trace"] + [{
                "agent": "paper_agent", **log_entry,
            }],
        }

    def web_agent_node(state: ResearchState) -> dict:
        store = EvidenceStore.from_dict(state["evidence"])
        budget = Budget.from_dict(state["budget"])
        source_texts = dict(state.get("source_texts", {}))

        try:
            t0 = time.perf_counter()
            result = run_web_agent(
                model=model,
                task=state["next_task"],
                evidence_store=store,
                max_tool_calls=MAX_WORKER_TOOL_CALLS,
                execute_fn=execute_fn,
            )
            latency = round(time.perf_counter() - t0, 2)

            budget.record_tool_calls(result["tool_calls"])
            budget.record_tokens(result["tokens_in"], result["tokens_out"])
            source_texts.update(result.get("source_texts", {}))

            log_entry = {
                "agent": "web_agent",
                "task": state["next_task"],
                "evidence_added": result["evidence_added"],
                "summary": result["summary"],
                "tool_calls": result["tool_calls"],
                "tool_errors": result["tool_errors"],
                "tokens_in": result["tokens_in"],
                "tokens_out": result["tokens_out"],
                "latency_s": latency,
            }
        except Exception as e:
            log_entry = {
                "agent": "web_agent",
                "task": state["next_task"],
                "evidence_added": [],
                "summary": f"Error: {str(e)[:200]}",
                "error": str(e)[:200],
            }

        return {
            "evidence": store.to_dict(),
            "worker_log": state["worker_log"] + [log_entry],
            "budget": budget.to_dict(),
            "source_texts": source_texts,
            "trace": state["trace"] + [{
                "agent": "web_agent", **log_entry,
            }],
        }

    def fact_checker_node(state: ResearchState) -> dict:
        store = EvidenceStore.from_dict(state["evidence"])
        budget = Budget.from_dict(state["budget"])
        source_texts = state.get("source_texts", {})

        try:
            t0 = time.perf_counter()
            result = run_fact_checker(model, store, source_texts)
            latency = round(time.perf_counter() - t0, 2)

            budget.record_tokens(result["total_tokens_in"], result["total_tokens_out"])

            log_entry = {
                "agent": "fact_checker",
                "task": state["next_task"],
                "summary": (
                    f"Checked {result['checked']}: "
                    f"{result['supported']} supported, "
                    f"{result['unsupported']} unsupported, "
                    f"{result['contradicted']} contradicted, "
                    f"{result['quote_not_found']} quote-not-found"
                ),
                "checked": result["checked"],
                "supported": result["supported"],
                "unsupported": result["unsupported"],
                "contradicted": result["contradicted"],
                "quote_not_found": result["quote_not_found"],
                "latency_s": latency,
            }
        except Exception as e:
            log_entry = {
                "agent": "fact_checker",
                "task": state["next_task"],
                "summary": f"Error: {str(e)[:200]}",
                "error": str(e)[:200],
            }

        return {
            "evidence": store.to_dict(),
            "worker_log": state["worker_log"] + [log_entry],
            "budget": budget.to_dict(),
            "trace": state["trace"] + [{
                "agent": "fact_checker", **log_entry,
            }],
        }

    def writer_node(state: ResearchState) -> dict:
        store = EvidenceStore.from_dict(state["evidence"])
        budget = Budget.from_dict(state["budget"])

        try:
            t0 = time.perf_counter()
            supported = store.supported()
            result = run_writer(model, state["question"], supported)
            latency = round(time.perf_counter() - t0, 2)

            if isinstance(result, dict) and "needs_more_evidence" in result:
                log_entry = {
                    "agent": "writer",
                    "task": state["next_task"],
                    "summary": f"Needs more evidence: {result['needs_more_evidence']}",
                    "needs_more_evidence": result["needs_more_evidence"],
                    "latency_s": latency,
                }
                report = ""
            elif isinstance(result, dict) and "report" in result:
                report = result["report"]
                budget.record_tokens(result["input_tokens"], result["output_tokens"])
                log_entry = {
                    "agent": "writer",
                    "task": state["next_task"],
                    "summary": f"Wrote report ({len(report)} chars) from {len(supported)} supported claims",
                    "report_length": len(report),
                    "citation_coverage": result["citation_coverage"],
                    "latency_s": latency,
                }
            else:
                # Legacy string return (v1 compat)
                report = result
                log_entry = {
                    "agent": "writer",
                    "task": state["next_task"],
                    "summary": f"Wrote report ({len(report)} chars) from {len(supported)} supported claims",
                    "report_length": len(report),
                    "latency_s": latency,
                }
        except Exception as e:
            report = ""
            log_entry = {
                "agent": "writer",
                "task": state["next_task"],
                "summary": f"Error: {str(e)[:200]}",
                "error": str(e)[:200],
            }

        return {
            "report": report,
            "worker_log": state["worker_log"] + [log_entry],
            "budget": budget.to_dict(),
            "trace": state["trace"] + [{
                "agent": "writer", **log_entry,
            }],
        }

    def approval_gate_node(state: ResearchState) -> dict:
        """Pause for human review before publishing the report."""
        store = EvidenceStore.from_dict(state["evidence"])
        decision = interrupt({
            "type": "approval_required",
            "report": state["report"],
            "question": state["question"],
            "supported_claims": len(store.supported()),
            "total_claims": len(store),
        })

        action = decision.get("action", "approve") if isinstance(decision, dict) else "approve"
        feedback = decision.get("feedback", "") if isinstance(decision, dict) else ""

        if action == "approve":
            return {
                "approval_status": "approved",
                "approval_feedback": "",
                "stop_reason": "finished",
                "trace": state["trace"] + [{
                    "agent": "approval_gate", "action": "approved",
                }],
            }
        else:
            # Rejected: clear report so supervisor doesn't auto-FINISH
            return {
                "approval_status": "rejected",
                "approval_feedback": feedback,
                "report": "",
                "stop_reason": "",
                "trace": state["trace"] + [{
                    "agent": "approval_gate", "action": "rejected",
                    "feedback": feedback,
                }],
            }

    def route_after_supervisor(state: ResearchState) -> str:
        agent = state.get("next_agent", "FINISH")
        if agent == "FINISH":
            # If a report exists, route to approval gate instead of ending
            if state.get("report"):
                return "approval_gate"
            return END
        if agent in ("paper_agent", "web_agent", "fact_checker", "writer"):
            return agent
        return END  # safety fallback

    def route_after_approval(state: ResearchState) -> str:
        if state.get("approval_status") == "approved":
            return END
        # Rejected — loop back to supervisor for revisions
        return "supervisor"

    # Wire the graph
    g = StateGraph(ResearchState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("paper_agent", paper_agent_node)
    g.add_node("web_agent", web_agent_node)
    g.add_node("fact_checker", fact_checker_node)
    g.add_node("writer", writer_node)
    g.add_node("approval_gate", approval_gate_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_after_supervisor)
    g.add_conditional_edges("approval_gate", route_after_approval)

    # Workers always return to supervisor
    for worker in ("paper_agent", "web_agent", "fact_checker", "writer"):
        g.add_edge(worker, "supervisor")

    checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return g.compile(checkpointer=checkpointer)


def _make_config(thread_id: str | None = None) -> dict:
    tid = thread_id or str(uuid4())
    return {
        "recursion_limit": 2 * MAX_SUPERVISOR_TURNS + 10,
        "configurable": {"thread_id": tid},
    }


def _format_result(state: dict, question: str, latency: float) -> dict[str, Any]:
    final_store = EvidenceStore.from_dict(state["evidence"])
    final_budget = Budget.from_dict(state["budget"])
    return {
        "question": question,
        "report": state["report"],
        "evidence": state["evidence"],
        "supported_claims": len(final_store.supported()),
        "total_claims": len(final_store),
        "supervisor_turns": state["supervisor_turns"],
        "stop_reason": state.get("stop_reason") or "finished",
        "approval_status": state.get("approval_status", ""),
        "budget": final_budget.to_dict(),
        "trace": state["trace"],
        "worker_log": state["worker_log"],
        "latency_s": latency,
    }


def _is_paused(graph, config: dict) -> bool:
    """Check if the graph is paused at an interrupt (approval gate)."""
    snapshot = graph.get_state(config)
    return bool(snapshot.next)  # non-empty means nodes are waiting to run


def run_research(graph, question: str, thread_id: str | None = None) -> dict[str, Any]:
    """Run the research pipeline. Pauses at the approval gate for human review.

    Returns a result dict. If approval_status is "pending", call
    resume_research() with the same graph and thread_id to approve or reject.
    """
    budget = Budget()
    store = EvidenceStore()
    t0 = time.perf_counter()

    config = _make_config(thread_id)
    thread_id = config["configurable"]["thread_id"]

    state = graph.invoke(
        {
            "question": question,
            "evidence": store.to_dict(),
            "next_agent": "",
            "next_task": "",
            "supervisor_turns": 0,
            "worker_log": [],
            "report": "",
            "budget": budget.to_dict(),
            "trace": [],
            "stop_reason": "",
            "source_texts": {},
            "approval_status": "",
            "approval_feedback": "",
        },
        config=config,
    )

    latency = round(time.perf_counter() - t0, 2)
    result = _format_result(state, question, latency)

    if _is_paused(graph, config):
        result["approval_status"] = "pending"

    result["thread_id"] = thread_id
    return result


def resume_research(graph, thread_id: str, action: str,
                    feedback: str = "") -> dict[str, Any]:
    """Resume the graph after human review.

    Args:
        graph: compiled graph (same instance used for run_research)
        thread_id: the thread_id from the paused run
        action: "approve" or "reject"
        feedback: reviewer feedback (used when action is "reject")

    Returns:
        Updated result dict.
    """
    config = _make_config(thread_id)
    t0 = time.perf_counter()

    state = graph.invoke(
        Command(resume={"action": action, "feedback": feedback}),
        config=config,
    )

    latency = round(time.perf_counter() - t0, 2)
    result = _format_result(state, state["question"], latency)

    if _is_paused(graph, config):
        result["approval_status"] = "pending"

    result["thread_id"] = thread_id
    return result
