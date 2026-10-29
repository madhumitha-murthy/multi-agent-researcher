# graph.py - LangGraph multi-agent orchestrator
#
# Flow: planner -> collectors (parallel) -> critic -> (back to collectors OR writer)
#
# The critic feedback loop is the key: if the critic finds gaps,
# it spawns new collectors to fill them. Max 2 rounds of feedback.

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.collector import run_collector
from agents.critic import run_critic
from agents.grounding import check_report_grounding, embedding_similarity
from agents.planner import run_planner
from agents.writer import run_writer
from config import MAX_CRITIC_ROUNDS


class ResearchState(TypedDict):
    question: str
    sub_tasks: list[str]
    papers: list[dict]
    critic_round: int
    critic_verdict: str
    gaps: list[str]
    report: str
    grounding: dict
    trace: list[dict]


def build_graph(model, similarity=embedding_similarity):
    """Build the multi-agent LangGraph."""

    def planner_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        sub_tasks = run_planner(model, state["question"])
        latency = round(time.perf_counter() - t0, 2)
        return {
            "sub_tasks": sub_tasks,
            "trace": state["trace"] + [{
                "agent": "planner", "sub_tasks": sub_tasks, "latency_s": latency,
            }],
        }

    def collectors_node(state: ResearchState) -> dict:
        """Run collectors in parallel using threads. Each gets its own sub-task."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_papers = list(state["papers"])
        events = []
        t0 = time.perf_counter()

        # Run collectors concurrently — each thread gets its own sub-task
        with ThreadPoolExecutor(max_workers=len(state["sub_tasks"])) as pool:
            futures = {
                pool.submit(run_collector, model, task): task
                for task in state["sub_tasks"]
            }
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                all_papers.extend(result["papers"])
                events.append({
                    "agent": "collector",
                    "sub_task": task,
                    "papers_found": len(result["papers"]),
                    "calls_used": result["calls_used"],
                    "trace": result["trace"],
                })

        # Deduplicate across all collectors
        seen = set()
        unique = []
        for p in all_papers:
            key = p["title"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        latency = round(time.perf_counter() - t0, 2)
        return {
            "papers": unique,
            "sub_tasks": [],  # clear sub_tasks after collection
            "trace": state["trace"] + [{
                "agent": "collectors_batch",
                "n_collectors": len(events),
                "total_papers": len(unique),
                "latency_s": latency,
                "details": events,
            }],
        }

    def critic_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        round_num = state["critic_round"] + 1
        result = run_critic(model, state["question"], state["papers"], round_num)
        latency = round(time.perf_counter() - t0, 2)

        return {
            "papers": result["papers"],
            "critic_round": round_num,
            "critic_verdict": result["verdict"],
            "gaps": result["gaps"],
            "trace": state["trace"] + [{
                "agent": "critic",
                "round": round_num,
                "verdict": result["verdict"],
                "gaps": result["gaps"],
                "papers_kept": len(result["papers"]),
                "reasoning": result["reasoning"],
                "latency_s": latency,
            }],
        }

    def gap_filler_node(state: ResearchState) -> dict:
        """Convert critic's gaps into new sub-tasks for collectors."""
        return {"sub_tasks": state["gaps"]}

    def writer_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        report = run_writer(model, state["question"], state["papers"])
        latency = round(time.perf_counter() - t0, 2)
        return {
            "report": report,
            "trace": state["trace"] + [{
                "agent": "writer",
                "report_length": len(report),
                "latency_s": latency,
            }],
        }

    def grounding_node(state: ResearchState) -> dict:
        """Check if the writer's report is actually supported by the papers."""
        t0 = time.perf_counter()
        verdict = check_report_grounding(state["report"], state["papers"], similarity)
        latency = round(time.perf_counter() - t0, 2)

        warning = ""
        if not verdict["grounded"]:
            warning = (
                "\n\n---\n**Note:** This report may contain claims not fully supported "
                "by the sources collected. Please verify key claims against the references."
            )

        return {
            "report": state["report"] + warning,
            "grounding": verdict,
            "trace": state["trace"] + [{
                "agent": "grounding", "latency_s": latency, **verdict,
            }],
        }

    def route_after_critic(state: ResearchState) -> str:
        """Critic says 'more_research' -> back to collectors. Otherwise -> writer."""
        if (
            state["critic_verdict"] == "more_research"
            and state["gaps"]
            and state["critic_round"] < MAX_CRITIC_ROUNDS
        ):
            return "gap_filler"
        return "writer"

    # Wire the graph
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("collectors", collectors_node)
    g.add_node("critic", critic_node)
    g.add_node("gap_filler", gap_filler_node)
    g.add_node("writer", writer_node)
    g.add_node("grounding", grounding_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "collectors")
    g.add_edge("collectors", "critic")
    g.add_conditional_edges("critic", route_after_critic, {
        "gap_filler": "gap_filler",
        "writer": "writer",
    })
    g.add_edge("gap_filler", "collectors")  # feedback loop
    g.add_edge("writer", "grounding")
    g.add_edge("grounding", END)

    return g.compile()


def run_research(graph, question: str) -> dict[str, Any]:
    """Run the full multi-agent research pipeline."""
    t0 = time.perf_counter()
    state = graph.invoke(
        {
            "question": question,
            "sub_tasks": [],
            "papers": [],
            "critic_round": 0,
            "critic_verdict": "",
            "gaps": [],
            "report": "",
            "grounding": {},
            "trace": [],
        },
        config={"recursion_limit": 30},
    )
    return {
        "question": question,
        "report": state["report"],
        "papers": state["papers"],
        "critic_rounds": state["critic_round"],
        "grounding": state["grounding"],
        "trace": state["trace"],
        "latency_s": round(time.perf_counter() - t0, 2),
    }
