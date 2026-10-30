# tests/test_workers.py - Tests for paper_agent, web_agent, react_loop

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.paper_agent import run_paper_agent
from agents.react_loop import run_react
from agents.web_agent import run_web_agent
from evidence import EvidenceStore
from llm import FakeModel, ModelResponse
from tools.url_reader import is_url_safe


# --- ReAct loop tests ---

def test_react_loop_basic():
    """ReAct loop executes tool calls and returns results."""
    fake_result = {"result": [{"title": "Paper A", "abstract": "About X", "url": "http://a"}]}

    model = FakeModel([
        ModelResponse(tool_calls=[{"name": "search_arxiv", "args": {"query": "test"}}]),
        ModelResponse(text="Done searching."),
    ])

    result = run_react(
        model=model,
        system_prompt="test",
        task="find papers",
        tool_schemas=[{"name": "search_arxiv", "description": "test", "params": {}}],
        max_tool_calls=3,
        execute_fn=lambda name, args: fake_result,
    )

    assert result["stats"]["calls_used"] == 1
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["tool"] == "search_arxiv"


def test_react_loop_budget_per_call():
    """Budget is checked per tool call, not per response."""
    call_count = {"n": 0}

    def counting_exec(name, args):
        call_count["n"] += 1
        return {"result": [{"title": f"P{call_count['n']}", "url": f"http://{call_count['n']}"}]}

    model = FakeModel([
        # Model tries to call 3 tools in one response
        ModelResponse(tool_calls=[
            {"name": "search_arxiv", "args": {"query": "a"}},
            {"name": "search_arxiv", "args": {"query": "b"}},
            {"name": "search_arxiv", "args": {"query": "c"}},
        ]),
        ModelResponse(text="Done."),
    ])

    result = run_react(
        model=model,
        system_prompt="test",
        task="find papers",
        tool_schemas=[{"name": "search_arxiv", "description": "test", "params": {}}],
        max_tool_calls=2,  # budget of 2
        execute_fn=counting_exec,
    )

    # Should stop after 2 calls, not execute the 3rd
    assert result["stats"]["calls_used"] == 2
    assert call_count["n"] == 2


def test_react_loop_tracks_tool_errors():
    model = FakeModel([
        ModelResponse(tool_calls=[{"name": "search_arxiv", "args": {"query": "test"}}]),
        ModelResponse(text="Done."),
    ])

    result = run_react(
        model=model,
        system_prompt="test",
        task="task",
        tool_schemas=[],
        max_tool_calls=3,
        execute_fn=lambda name, args: {"error": "API failed"},
    )

    assert result["stats"]["tool_errors"] == 1


def test_react_loop_tracks_seen_urls():
    """URLs from tool results are added to seen_urls for read_url allowlist."""
    fake_result = {"result": [
        {"title": "P1", "url": "http://found-url.com"},
        {"title": "P2", "url": "http://another.com"},
    ]}

    model = FakeModel([
        ModelResponse(tool_calls=[{"name": "search_web", "args": {"query": "test"}}]),
        ModelResponse(text="Done."),
    ])

    result = run_react(
        model=model,
        system_prompt="test",
        task="task",
        tool_schemas=[],
        max_tool_calls=3,
        execute_fn=lambda name, args: fake_result,
        allowed_urls=set(),
    )

    assert "http://found-url.com" in result["seen_urls"]
    assert "http://another.com" in result["seen_urls"]


# --- Paper agent tests ---

def test_paper_agent_adds_valid_evidence():
    """Paper agent adds evidence when quote matches source text."""
    store = EvidenceStore()

    search_result = {"result": [
        {"title": "Hallucination Survey",
         "abstract": "LLMs often produce incorrect outputs known as hallucinations",
         "url": "http://paper1", "source": "semantic_scholar"},
    ]}

    extraction_json = json.dumps([{
        "claim": "LLMs produce incorrect outputs",
        "source_title": "Hallucination Survey",
        "source_url": "http://paper1",
        "source_type": "semantic_scholar",
        "quote": "LLMs often produce incorrect outputs known as hallucinations",
    }])

    model = FakeModel([
        # ReAct: tool call + final answer
        ModelResponse(tool_calls=[{"name": "search_semantic_scholar", "args": {"query": "hallucinations"}}]),
        ModelResponse(text="Found a survey."),
        # Extraction call
        ModelResponse(text=extraction_json),
    ])

    result = run_paper_agent(
        model=model,
        task="find hallucination papers",
        evidence_store=store,
        max_tool_calls=3,
        execute_fn=lambda name, args: search_result,
    )

    assert len(result["evidence_added"]) == 1
    assert result["extraction_rejected"] == 0
    assert store.get("E1").claim == "LLMs produce incorrect outputs"
    assert store.get("E1").found_by == "paper_agent"


def test_paper_agent_rejects_fabricated_quote():
    """Fabricated quotes (not in source text) are rejected."""
    store = EvidenceStore()

    search_result = {"result": [
        {"title": "Real Paper",
         "abstract": "This paper studies reinforcement learning for robotics",
         "url": "http://paper1", "source": "arxiv"},
    ]}

    extraction_json = json.dumps([{
        "claim": "RL achieves 99% success rate on all tasks",
        "source_title": "Real Paper",
        "source_url": "http://paper1",
        "source_type": "arxiv",
        "quote": "RL achieves 99% success rate on all benchmarks",  # FABRICATED
    }])

    model = FakeModel([
        ModelResponse(tool_calls=[{"name": "search_arxiv", "args": {"query": "RL"}}]),
        ModelResponse(text="Done."),
        ModelResponse(text=extraction_json),
    ])

    result = run_paper_agent(
        model=model,
        task="find RL papers",
        evidence_store=store,
        max_tool_calls=3,
        execute_fn=lambda name, args: search_result,
    )

    assert len(result["evidence_added"]) == 0
    assert result["extraction_rejected"] == 1
    assert len(store) == 0


def test_paper_agent_rejects_unknown_url():
    """Claims citing a URL not returned by any tool are rejected."""
    store = EvidenceStore()

    search_result = {"result": [
        {"title": "Real Paper", "abstract": "About X", "url": "http://real", "source": "arxiv"},
    ]}

    extraction_json = json.dumps([{
        "claim": "claim",
        "source_title": "Fake Paper",
        "source_url": "http://invented-url",
        "source_type": "arxiv",
        "quote": "About X",
    }])

    model = FakeModel([
        ModelResponse(tool_calls=[{"name": "search_arxiv", "args": {"query": "X"}}]),
        ModelResponse(text="Done."),
        ModelResponse(text=extraction_json),
    ])

    result = run_paper_agent(
        model=model,
        task="task",
        evidence_store=store,
        max_tool_calls=3,
        execute_fn=lambda name, args: search_result,
    )

    assert len(result["evidence_added"]) == 0
    assert result["extraction_rejected"] == 1


# --- Web agent tests ---

def test_web_agent_adds_valid_evidence():
    store = EvidenceStore()

    search_result = {"result": [
        {"title": "LangChain Blog",
         "abstract": "LangChain provides tools for building RAG applications",
         "url": "http://blog1", "source": "web"},
    ]}

    extraction_json = json.dumps([{
        "claim": "LangChain provides RAG tools",
        "source_title": "LangChain Blog",
        "source_url": "http://blog1",
        "source_type": "web",
        "quote": "LangChain provides tools for building RAG applications",
    }])

    model = FakeModel([
        ModelResponse(tool_calls=[{"name": "search_web", "args": {"query": "RAG"}}]),
        ModelResponse(text="Found blog."),
        ModelResponse(text=extraction_json),
    ])

    result = run_web_agent(
        model=model,
        task="find RAG implementations",
        evidence_store=store,
        max_tool_calls=3,
        execute_fn=lambda name, args: search_result,
    )

    assert len(result["evidence_added"]) == 1
    assert store.get("E1").found_by == "web_agent"


def test_web_agent_rejects_fabricated_quote():
    store = EvidenceStore()

    search_result = {"result": [
        {"title": "Blog Post", "abstract": "An overview of modern NLP techniques",
         "url": "http://blog", "source": "web"},
    ]}

    extraction_json = json.dumps([{
        "claim": "GPT-5 outperforms all models",
        "source_title": "Blog Post",
        "source_url": "http://blog",
        "source_type": "web",
        "quote": "GPT-5 has been shown to outperform all existing models",  # FABRICATED
    }])

    model = FakeModel([
        ModelResponse(tool_calls=[{"name": "search_web", "args": {"query": "NLP"}}]),
        ModelResponse(text="Done."),
        ModelResponse(text=extraction_json),
    ])

    result = run_web_agent(
        model=model,
        task="task",
        evidence_store=store,
        max_tool_calls=3,
        execute_fn=lambda name, args: search_result,
    )

    assert len(result["evidence_added"]) == 0
    assert result["extraction_rejected"] == 1


# --- URL safety tests ---

def test_read_url_blocks_private_addresses():
    safe, _ = is_url_safe("http://localhost/secret")
    assert safe is False

    safe, _ = is_url_safe("http://127.0.0.1/admin")
    assert safe is False

    safe, _ = is_url_safe("http://192.168.1.1/config")
    assert safe is False

    safe, _ = is_url_safe("http://10.0.0.1/internal")
    assert safe is False


def test_read_url_blocks_non_http():
    safe, _ = is_url_safe("file:///etc/passwd")
    assert safe is False

    safe, _ = is_url_safe("ftp://example.com/file")
    assert safe is False


def test_read_url_allows_https():
    safe, _ = is_url_safe("https://arxiv.org/abs/1234")
    assert safe is True


def test_read_url_rejects_unseen_url():
    """read_url with allowed_urls rejects URLs not in the set."""
    from tools.url_reader import read_url
    result = read_url("http://unseen.com/page", allowed_urls={"http://allowed.com"})
    assert "error" in result
    assert "not in search results" in result["error"]


# Need json import for the extraction JSON
import json
