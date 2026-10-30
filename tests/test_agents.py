# test_agents.py - Tests for all agents + orchestrator using FakeModel

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from baselines.pipeline.collector import run_collector
from baselines.pipeline.critic import run_critic
from baselines.pipeline.grounding import check_report_grounding, split_sentences
from baselines.pipeline.planner import run_planner
from baselines.pipeline.writer import run_writer
from eval.run_eval import compute_metrics
from llm import FakeModel, ModelResponse, strip_json_fences
from tools.registry import execute_tool

# --- Fake similarity for graph tests ---

def high_sim(sentences, sources):
    """Returns high similarity scores so grounding always passes."""
    return [[0.9] * len(sources) for _ in sentences]


def low_sim(sentences, sources):
    """Returns low similarity scores so grounding always fails."""
    return [[0.1] * len(sources) for _ in sentences]


# --- JSON fence stripping tests ---

def test_strip_json_fences_with_fences():
    raw = '```json\n{"sub_tasks": ["a", "b"]}\n```'
    assert strip_json_fences(raw) == '{"sub_tasks": ["a", "b"]}'


def test_strip_json_fences_without_fences():
    raw = '{"sub_tasks": ["a", "b"]}'
    assert strip_json_fences(raw) == '{"sub_tasks": ["a", "b"]}'


def test_planner_handles_fenced_json():
    model = FakeModel([
        ModelResponse(text='```json\n{"sub_tasks": ["topic A", "topic B"]}\n```')
    ])
    tasks = run_planner(model, "some question")
    assert tasks == ["topic A", "topic B"]


def test_critic_handles_fenced_json():
    papers = [{"title": "P1", "source": "arxiv", "year": 2024, "abstract": "About X"}]
    model = FakeModel([
        ModelResponse(text='```json\n{"verdict": "more_research", "gaps": ["missing Y"], "keep_papers": [1], "reasoning": "Need Y"}\n```')
    ])
    result = run_critic(model, "q", papers, round_num=1)
    assert result["verdict"] == "more_research"
    assert "missing Y" in result["gaps"]


# --- Planner tests ---

def test_planner_returns_sub_tasks():
    model = FakeModel([
        ModelResponse(text='{"sub_tasks": ["hallucination detection", "hallucination mitigation"]}')
    ])
    tasks = run_planner(model, "How to reduce hallucinations in LLMs?")
    assert tasks == ["hallucination detection", "hallucination mitigation"]


def test_planner_fallback_on_bad_json():
    model = FakeModel([ModelResponse(text="not json at all")])
    tasks = run_planner(model, "some question")
    assert tasks == ["some question"]  # falls back to original question


def test_planner_caps_at_3_tasks():
    model = FakeModel([
        ModelResponse(text='{"sub_tasks": ["a", "b", "c", "d", "e"]}')
    ])
    tasks = run_planner(model, "big question")
    assert len(tasks) == 3


# --- Collector tests ---

def test_collector_calls_tools_and_collects_papers():
    fake_papers = [{"title": "Paper A", "abstract": "About X", "url": "http://a", "source": "arxiv"}]
    model = FakeModel([
        ModelResponse(tool_calls=[{"name": "search_arxiv", "args": {"query": "test"}}]),
        ModelResponse(text="Found 1 paper about test."),
    ])

    import tools.registry as reg
    original = reg.execute_tool
    reg.execute_tool = lambda name, args: {"result": fake_papers}

    try:
        result = run_collector(model, "test topic")
        assert len(result["papers"]) == 1
        assert result["papers"][0]["title"] == "Paper A"
        assert result["calls_used"] == 1
    finally:
        reg.execute_tool = original


def test_collector_deduplicates_papers():
    dup_papers = [
        {"title": "Same Paper", "abstract": "v1", "url": "http://a", "source": "arxiv"},
        {"title": "Same Paper", "abstract": "v2", "url": "http://b", "source": "s2"},
    ]
    model = FakeModel([
        ModelResponse(tool_calls=[{"name": "search_arxiv", "args": {"query": "q"}}]),
        ModelResponse(text="Done."),
    ])

    import tools.registry as reg
    original = reg.execute_tool
    reg.execute_tool = lambda name, args: {"result": dup_papers}

    try:
        result = run_collector(model, "q")
        assert len(result["papers"]) == 1
    finally:
        reg.execute_tool = original


# --- Critic tests ---

def test_critic_approves_good_coverage():
    papers = [{"title": f"Paper {i}", "source": "arxiv", "year": 2024, "abstract": f"About topic {i}"} for i in range(6)]
    model = FakeModel([
        ModelResponse(text='{"verdict": "approve", "gaps": [], "keep_papers": [1,2,3,4,5,6], "reasoning": "Good coverage"}')
    ])
    result = run_critic(model, "test question", papers, round_num=1)
    assert result["verdict"] == "approve"
    assert len(result["papers"]) == 6
    assert result["gaps"] == []


def test_critic_requests_more_research():
    papers = [{"title": "Paper A", "source": "arxiv", "year": 2024, "abstract": "Only about detection"}]
    model = FakeModel([
        ModelResponse(text='{"verdict": "more_research", "gaps": ["mitigation methods"], "keep_papers": [1], "reasoning": "Missing mitigation"}')
    ])
    result = run_critic(model, "test question", papers, round_num=1)
    assert result["verdict"] == "more_research"
    assert "mitigation methods" in result["gaps"]


def test_critic_handles_bad_json():
    papers = [{"title": "P1", "source": "arxiv", "year": 2024, "abstract": "..."}]
    model = FakeModel([ModelResponse(text="not json")])
    result = run_critic(model, "q", papers, round_num=1)
    assert result["verdict"] == "approve"  # fallback
    assert len(result["papers"]) == 1


# --- Writer tests ---

def test_writer_produces_report_v1_compat():
    """v1 compatibility: writer accepts paper dicts."""
    papers = [
        {"title": "Paper A", "abstract": "Method A for X", "url": "http://a", "year": 2024, "source": "arxiv", "authors": "Smith"},
        {"title": "Paper B", "abstract": "Method B for X", "url": "http://b", "year": 2024, "source": "s2", "authors": "Jones"},
    ]
    model = FakeModel([
        ModelResponse(text="## Overview\nThis report covers...\n\n## Key Findings\n- Finding 1 [Paper A]\n\n## References\n1. Paper A")
    ])
    report = run_writer(model, "test question", papers)
    assert "Overview" in report
    assert len(report) > 50


# --- Grounding tests ---

def test_grounding_passes_with_high_similarity():
    report = "This paper proposes a new method for hallucination detection. The approach uses retrieval augmented generation."
    papers = [{"title": "Hallucination Detection", "abstract": "A new method for hallucination detection using RAG."}]
    result = check_report_grounding(report, papers, similarity=high_sim)
    assert result["grounded"] is True
    assert result["supported_fraction"] > 0.5


def test_grounding_fails_with_low_similarity():
    report = "This paper proposes a new method for hallucination detection. The approach uses retrieval augmented generation."
    papers = [{"title": "Unrelated Paper", "abstract": "Something completely different about cooking."}]
    result = check_report_grounding(report, papers, similarity=low_sim)
    assert result["grounded"] is False
    assert result["supported_fraction"] < 0.5


def test_grounding_no_sources():
    result = check_report_grounding("Some report text here.", [], similarity=high_sim)
    assert result["grounded"] is False
    assert result["reason"] == "no sources"


def test_split_sentences():
    text = "This is a longer sentence about machine learning models. Another sentence about natural language processing techniques! A third sentence covering deep learning approaches?"
    sents = split_sentences(text)
    assert len(sents) >= 2


def test_split_sentences_strips_references():
    text = (
        "## Overview\nThis report covers hallucination detection methods.\n\n"
        "## References\n1. Hallucination Detection Paper (2024)\n2. RAG Survey (2023)\n"
    )
    sents = split_sentences(text)
    # Reference lines should be stripped — none should appear
    for s in sents:
        assert "Hallucination Detection Paper" not in s
        assert "RAG Survey" not in s


# --- Tool registry tests ---

def test_unknown_tool_returns_error():
    result = execute_tool("nonexistent_tool", {})
    assert "error" in result


# --- Graph integration test (v1 pipeline, now in baselines) ---

def test_full_graph_runs():
    from baselines.pipeline.graph import build_graph, run_research

    fake_papers = [{"title": "P1", "abstract": "About X", "url": "http://p1", "source": "arxiv"}]

    model = FakeModel([
        # Planner
        ModelResponse(text='{"sub_tasks": ["topic A"]}'),
        # Collector: tool call then answer
        ModelResponse(tool_calls=[{"name": "search_arxiv", "args": {"query": "topic A"}}]),
        ModelResponse(text="Found papers."),
        # Critic: approve
        ModelResponse(text='{"verdict": "approve", "gaps": [], "keep_papers": [1], "reasoning": "ok"}'),
        # Writer - needs sentences > 25 chars for grounding to check
        ModelResponse(text="## Overview\nThis report provides a comprehensive analysis of topic A and its applications. The research covers multiple approaches to solving the core problem. Several papers have been reviewed and synthesised below.\n\n## References\n1. P1"),
    ])

    import tools.registry as reg
    original = reg.execute_tool
    reg.execute_tool = lambda name, args: {"result": fake_papers}

    try:
        graph = build_graph(model, similarity=high_sim)
        result = run_research(graph, "test question")
        assert result["report"] != ""
        assert len(result["papers"]) >= 1
        # Check all agents ran
        agents_used = [e["agent"] for e in result["trace"]]
        assert "planner" in agents_used
        assert "collectors_batch" in agents_used
        assert "critic" in agents_used
        assert "writer" in agents_used
        assert "grounding" in agents_used
        # Grounding should pass with high_sim
        assert result["grounding"]["grounded"] is True
    finally:
        reg.execute_tool = original


# --- Graph with critic feedback loop (v1 pipeline) ---

def test_critic_feedback_loop():
    from baselines.pipeline.graph import build_graph, run_research

    paper_a = {"title": "Paper A", "abstract": "Detection", "url": "http://a", "source": "arxiv"}
    paper_b = {"title": "Paper B", "abstract": "Mitigation", "url": "http://b", "source": "arxiv"}
    call_count = {"n": 0}

    def fake_execute(name, args):
        call_count["n"] += 1
        if call_count["n"] <= 1:
            return {"result": [paper_a]}
        return {"result": [paper_b]}

    model = FakeModel([
        # Planner
        ModelResponse(text='{"sub_tasks": ["detection methods"]}'),
        # Collector round 1
        ModelResponse(tool_calls=[{"name": "search_arxiv", "args": {"query": "detection"}}]),
        ModelResponse(text="Found detection papers."),
        # Critic round 1: needs more
        ModelResponse(text='{"verdict": "more_research", "gaps": ["mitigation methods"], "keep_papers": [1], "reasoning": "Missing mitigation"}'),
        # Collector round 2 (gap filler)
        ModelResponse(tool_calls=[{"name": "search_arxiv", "args": {"query": "mitigation"}}]),
        ModelResponse(text="Found mitigation papers."),
        # Critic round 2: approve
        ModelResponse(text='{"verdict": "approve", "gaps": [], "keep_papers": [1, 2], "reasoning": "Good now"}'),
        # Writer
        ModelResponse(text="## Overview\nReport covering both detection and mitigation."),
    ])

    import tools.registry as reg
    original = reg.execute_tool
    reg.execute_tool = fake_execute

    try:
        graph = build_graph(model, similarity=high_sim)
        result = run_research(graph, "hallucination approaches")
        assert result["critic_rounds"] == 2
        assert len(result["papers"]) == 2
        # Verify the loop happened
        critic_events = [e for e in result["trace"] if e["agent"] == "critic"]
        assert len(critic_events) == 2
        assert critic_events[0]["verdict"] == "more_research"
        assert critic_events[1]["verdict"] == "approve"
        # Grounding should pass
        assert result["grounding"]["grounded"] is True
    finally:
        reg.execute_tool = original


# --- Eval metrics test (v2) ---

def test_compute_metrics():
    records = [
        {"supported_claims": 5, "total_claims": 7, "supervisor_turns": 6,
         "latency_s": 30, "report_length": 500, "stop_reason": "finished",
         "cost_usd": 0.05, "citation_coverage": 0.8, "error": None},
        {"supported_claims": 3, "total_claims": 4, "supervisor_turns": 4,
         "latency_s": 25, "report_length": 400, "stop_reason": "finished",
         "cost_usd": 0.03, "citation_coverage": 0.6, "error": None},
        {"supported_claims": 0, "total_claims": 0, "supervisor_turns": 2,
         "latency_s": 10, "report_length": 100, "stop_reason": "finished",
         "cost_usd": 0.01, "citation_coverage": None, "error": None},
    ]
    m = compute_metrics(records)
    assert m["questions"] == 3
    assert m["completed"] == 3
    assert m["claims_found_rate"] == 0.667
    assert m["avg_supported_claims"] == 2.7
    assert m["avg_supervisor_turns"] == 4.0
