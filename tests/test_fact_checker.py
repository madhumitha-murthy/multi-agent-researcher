# tests/test_fact_checker.py - Tests for the fact-checker agent

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.fact_checker import (
    normalise_for_match,
    quote_in_source,
    run_fact_checker,
)
from evidence import EvidenceStore
from llm import FakeModel, ModelResponse

# --- Normalisation helpers ---

def test_normalise_collapses_whitespace():
    assert normalise_for_match("  hello   world  ") == "hello world"


def test_normalise_lowercases():
    assert normalise_for_match("Hello World") == "hello world"


# --- Quote matching ---

def test_quote_in_source_exact_match():
    assert quote_in_source("hello world", "The text says hello world here.")


def test_quote_in_source_case_insensitive():
    assert quote_in_source("Hello World", "the text says hello world here.")


def test_quote_in_source_whitespace_normalised():
    assert quote_in_source("hello  world", "the text says hello world here.")


def test_quote_not_in_source():
    assert not quote_in_source("completely different text", "hello world paper")


def test_quote_empty():
    assert not quote_in_source("", "some text")
    assert not quote_in_source("quote", "")


# --- Fact-checker: deterministic quote check ---

def _make_store_and_sources():
    """Helper: store with 2 items, source_texts for both."""
    store = EvidenceStore()
    store.add(
        claim="LLMs produce incorrect outputs called hallucinations",
        source_title="Paper A",
        source_url="http://a",
        source_type="arxiv",
        quote="often produce plausible yet incorrect outputs",
        found_by="paper_agent",
        task="task",
    )
    store.add(
        claim="RAG reduces hallucination rates by 50%",
        source_title="Paper B",
        source_url="http://b",
        source_type="semantic_scholar",
        quote="This quote does not exist in the source",
        found_by="paper_agent",
        task="task",
    )
    source_texts = {
        "http://a": "Large language models often produce plausible yet incorrect outputs known as hallucinations.",
        "http://b": "Retrieval augmented generation can help improve factual accuracy.",
    }
    return store, source_texts


def test_quote_not_in_source_marked_unsupported():
    """Core acceptance criterion: quote-not-in-source is ALWAYS unsupported."""
    store, source_texts = _make_store_and_sources()

    # LLM will only be called for E1 (quote matches); E2 fails deterministic check
    model = FakeModel([
        ModelResponse(text='[{"id": "E1", "verdict": "supported", "reason": "quote matches claim"}]')
    ])

    result = run_fact_checker(model, store, source_texts)

    assert result["quote_not_found"] == 1
    assert store.get("E2").status == "unsupported"
    assert "quote not found" in store.get("E2").checker_note


def test_quote_found_passes_to_llm():
    store, source_texts = _make_store_and_sources()

    model = FakeModel([
        ModelResponse(text='[{"id": "E1", "verdict": "supported", "reason": "matches"}]')
    ])

    run_fact_checker(model, store, source_texts)

    assert store.get("E1").status == "supported"
    # Verify the LLM was actually called
    assert len(model.calls) == 1


# --- Fact-checker: LLM entailment verdicts ---

def test_llm_supported_verdict():
    store = EvidenceStore()
    store.add("claim A", "Paper", "http://a", "arxiv",
              "exact quote from source", "paper_agent", "task")
    source_texts = {"http://a": "This paper says exact quote from source and more."}

    model = FakeModel([
        ModelResponse(text='[{"id": "E1", "verdict": "supported", "reason": "quote justifies claim"}]')
    ])
    run_fact_checker(model, store, source_texts)
    assert store.get("E1").status == "supported"


def test_llm_unsupported_verdict():
    store = EvidenceStore()
    store.add("claim A", "Paper", "http://a", "arxiv",
              "some quote", "paper_agent", "task")
    source_texts = {"http://a": "The paper contains some quote and other text."}

    model = FakeModel([
        ModelResponse(text='[{"id": "E1", "verdict": "unsupported", "reason": "claim adds info not in quote"}]')
    ])
    run_fact_checker(model, store, source_texts)
    assert store.get("E1").status == "unsupported"
    assert "adds info" in store.get("E1").checker_note


def test_llm_contradicted_verdict():
    store = EvidenceStore()
    store.add("claim A", "Paper", "http://a", "arxiv",
              "the method failed", "paper_agent", "task")
    source_texts = {"http://a": "We found that the method failed on all benchmarks."}

    model = FakeModel([
        ModelResponse(text='[{"id": "E1", "verdict": "contradicted", "reason": "source says opposite"}]')
    ])
    run_fact_checker(model, store, source_texts)
    assert store.get("E1").status == "contradicted"


# --- Batching ---

def test_batching_multiple_items():
    store = EvidenceStore()
    source_texts = {}
    claims = [
        ("Transformers use self-attention mechanisms for sequence modeling", "transformers use self-attention"),
        ("RLHF aligns language models with human preferences", "RLHF aligns language models"),
        ("Diffusion models generate high-quality images from noise", "diffusion models generate high-quality images"),
    ]
    for i, (claim, quote) in enumerate(claims):
        url = f"http://paper{i}"
        store.add(claim, f"Paper {i}", url, "arxiv", quote, "paper_agent", "task")
        source_texts[url] = f"The paper shows that {quote} and discusses further implications."

    # All 3 pass deterministic check, so all go to LLM in one batch
    model = FakeModel([
        ModelResponse(text='[{"id": "E1", "verdict": "supported", "reason": "ok"}, '
                           '{"id": "E2", "verdict": "unsupported", "reason": "weak"}, '
                           '{"id": "E3", "verdict": "supported", "reason": "ok"}]')
    ])
    result = run_fact_checker(model, store, source_texts)

    assert result["checked"] == 3
    assert result["supported"] == 2
    assert result["unsupported"] == 1
    assert store.get("E1").status == "supported"
    assert store.get("E2").status == "unsupported"
    assert store.get("E3").status == "supported"
    # Only 1 LLM call (all fit in one batch)
    assert len(model.calls) == 1


# --- Edge cases ---

def test_no_unverified_items():
    store = EvidenceStore()
    model = FakeModel([])  # should not be called
    result = run_fact_checker(model, store, {})
    assert result["checked"] == 0
    assert len(model.calls) == 0


def test_llm_bad_json_marks_unsupported():
    store = EvidenceStore()
    store.add("claim", "Paper", "http://a", "arxiv",
              "verbatim quote", "paper_agent", "task")
    source_texts = {"http://a": "Source contains verbatim quote here."}

    model = FakeModel([
        ModelResponse(text="not valid json at all")
    ])
    run_fact_checker(model, store, source_texts)
    assert store.get("E1").status == "unsupported"
    assert "parsing failed" in store.get("E1").checker_note


def test_missing_source_text_marks_unsupported():
    """If source_texts doesn't have the URL, quote check fails."""
    store = EvidenceStore()
    store.add("claim", "Paper", "http://missing", "arxiv",
              "some quote", "paper_agent", "task")

    model = FakeModel([])  # should not be called
    result = run_fact_checker(model, store, {})
    assert store.get("E1").status == "unsupported"
    assert result["quote_not_found"] == 1


def test_already_verified_items_skipped():
    store = EvidenceStore()
    store.add("claim", "Paper", "http://a", "arxiv",
              "quote text", "paper_agent", "task")
    store.set_status("E1", "supported", "already checked")

    model = FakeModel([])
    result = run_fact_checker(model, store, {"http://a": "quote text here"})
    assert result["checked"] == 0
    assert len(model.calls) == 0


def test_invalid_verdict_defaults_to_unsupported():
    store = EvidenceStore()
    store.add("claim", "Paper", "http://a", "arxiv",
              "valid quote", "paper_agent", "task")
    source_texts = {"http://a": "Contains valid quote in the text."}

    model = FakeModel([
        ModelResponse(text='[{"id": "E1", "verdict": "maybe", "reason": "unclear"}]')
    ])
    run_fact_checker(model, store, source_texts)
    assert store.get("E1").status == "unsupported"
