# tests/test_supervisor.py - Tests for supervisor agent routing

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.supervisor import _apply_guards, run_supervisor
from budget import Budget
from evidence import EvidenceStore
from llm import FakeModel, ModelResponse


def _empty_store():
    return EvidenceStore()


def _store_with_unverified():
    store = EvidenceStore()
    store.add("claim A", "Paper A", "http://a", "arxiv", "quote", "paper_agent", "task")
    return store


def _store_with_supported():
    store = EvidenceStore()
    store.add("claim A", "Paper A", "http://a", "arxiv", "quote", "paper_agent", "task")
    store.set_status("E1", "supported", "verified")
    return store


# --- Valid routing ---

def test_supervisor_routes_to_paper_agent():
    model = FakeModel([
        ModelResponse(text='{"next_agent": "paper_agent", "task": "find RL papers", "reasoning": "need academic sources"}')
    ])
    result = run_supervisor(model, "test question", _empty_store(), [], Budget())
    assert result["next_agent"] == "paper_agent"
    assert result["task"] == "find RL papers"


def test_supervisor_routes_to_web_agent():
    model = FakeModel([
        ModelResponse(text='{"next_agent": "web_agent", "task": "find implementations", "reasoning": "need practical sources"}')
    ])
    result = run_supervisor(model, "test question", _empty_store(), [], Budget())
    assert result["next_agent"] == "web_agent"


def test_supervisor_routes_to_fact_checker():
    model = FakeModel([
        ModelResponse(text='{"next_agent": "fact_checker", "task": "verify claims", "reasoning": "unverified claims exist"}')
    ])
    result = run_supervisor(model, "q", _store_with_unverified(), [], Budget())
    assert result["next_agent"] == "fact_checker"


def test_supervisor_routes_to_writer():
    model = FakeModel([
        ModelResponse(text='{"next_agent": "writer", "task": "write report", "reasoning": "enough evidence"}')
    ])
    result = run_supervisor(model, "q", _store_with_supported(), [], Budget())
    assert result["next_agent"] == "writer"


def test_supervisor_routes_to_finish():
    model = FakeModel([
        ModelResponse(text='{"next_agent": "FINISH", "task": "", "reasoning": "report is done"}')
    ])
    result = run_supervisor(model, "q", _store_with_supported(), [], Budget())
    assert result["next_agent"] == "FINISH"


# --- Invalid JSON fallback ---

def test_supervisor_fallback_on_bad_json_with_unverified():
    model = FakeModel([ModelResponse(text="not json at all")])
    result = run_supervisor(model, "q", _store_with_unverified(), [], Budget())
    assert result["next_agent"] == "fact_checker"


def test_supervisor_fallback_on_bad_json_with_supported():
    model = FakeModel([ModelResponse(text="garbage")])
    store = _store_with_supported()
    # No unverified items
    result = run_supervisor(model, "q", store, [], Budget())
    assert result["next_agent"] == "writer"


def test_supervisor_fallback_on_bad_json_empty_store():
    model = FakeModel([ModelResponse(text="???")])
    result = run_supervisor(model, "q", _empty_store(), [], Budget())
    assert result["next_agent"] == "FINISH"


def test_supervisor_fallback_on_invalid_agent_name():
    model = FakeModel([
        ModelResponse(text='{"next_agent": "nonexistent_agent", "task": "x", "reasoning": "y"}')
    ])
    result = run_supervisor(model, "q", _store_with_unverified(), [], Budget())
    # Should fall back, not crash
    assert result["next_agent"] in ("fact_checker", "writer", "FINISH", "paper_agent")


# --- Guard: cannot call writer with zero supported claims ---

def test_guard_writer_no_supported_redirects_to_fact_checker():
    agent, task, reason = _apply_guards(
        "writer", "write report", "time to write",
        _store_with_unverified(), Budget()
    )
    assert agent == "fact_checker"
    assert "no supported" in reason


def test_guard_writer_no_evidence_redirects_to_paper_agent():
    agent, task, reason = _apply_guards(
        "writer", "write report", "time to write",
        _empty_store(), Budget()
    )
    assert agent == "paper_agent"


# --- Guard: budget exhausted ---

def test_guard_budget_exhausted_redirects_to_writer():
    budget = Budget(max_turns=1)
    budget.record_turn()  # now exhausted

    agent, task, reason = _apply_guards(
        "paper_agent", "search more", "need more papers",
        _store_with_supported(), budget
    )
    assert agent == "writer"
    assert "turn_limit" in reason


def test_guard_budget_exhausted_no_evidence_finishes():
    budget = Budget(max_turns=1)
    budget.record_turn()

    agent, task, reason = _apply_guards(
        "paper_agent", "search", "searching",
        _empty_store(), budget
    )
    assert agent == "FINISH"


# --- Token tracking ---

def test_supervisor_returns_token_counts():
    model = FakeModel([
        ModelResponse(text='{"next_agent": "FINISH", "task": "", "reasoning": "done"}',
                      input_tokens=100, output_tokens=50)
    ])
    result = run_supervisor(model, "q", _empty_store(), [], Budget())
    assert result["tokens_in"] == 100
    assert result["tokens_out"] == 50
