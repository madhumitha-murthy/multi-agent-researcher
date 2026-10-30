# tests/test_budget.py - Tests for budget tracking

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from budget import Budget


def test_budget_initial_state():
    b = Budget()
    assert b.turns_used == 0
    assert b.cost_usd == 0.0
    assert b.exhausted() is None


def test_budget_turn_limit():
    b = Budget(max_turns=3)
    b.record_turn()
    b.record_turn()
    assert b.exhausted() is None
    b.record_turn()
    assert b.exhausted() == "turn_limit"


def test_budget_cost_limit():
    b = Budget(max_cost_usd=0.001)
    # Record enough tokens to exceed the cost limit
    b.record_tokens(100_000, 100_000)  # should cost more than $0.001
    assert b.exhausted() == "cost_limit"


def test_budget_cost_calculation():
    b = Budget()
    b.record_tokens(1_000_000, 0)  # 1M input tokens
    # With default PRICE_PER_1M_INPUT = 0.75 (Gemini 3.8 Flash)
    assert abs(b.cost_usd - 0.75) < 0.001


def test_budget_remaining_summary():
    b = Budget(max_turns=10, max_cost_usd=0.50)
    b.record_turn()
    b.record_tokens(100, 50)
    summary = b.remaining_summary()
    assert "1/10" in summary
    assert "$" in summary


def test_budget_serialisation():
    b = Budget(max_turns=10, max_cost_usd=0.50)
    b.record_turn()
    b.record_turn()
    b.record_tokens(500, 200)
    b.record_tool_calls(3)

    data = b.to_dict()
    restored = Budget.from_dict(data)

    assert restored.turns_used == 2
    assert restored.tool_calls_used == 3
    assert restored.tokens_in == 500
    assert restored.tokens_out == 200
    assert restored.max_turns == 10
