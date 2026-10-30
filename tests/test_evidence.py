# tests/test_evidence.py - Tests for Evidence dataclass + EvidenceStore

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evidence import Evidence, EvidenceStore


def _make_store_with_items():
    store = EvidenceStore()
    store.add(
        claim="LLMs hallucinate facts",
        source_title="Paper A",
        source_url="http://a",
        source_type="arxiv",
        quote="LLMs often produce plausible yet incorrect outputs",
        found_by="paper_agent",
        task="find hallucination papers",
    )
    store.add(
        claim="RAG reduces hallucination",
        source_title="Paper B",
        source_url="http://b",
        source_type="semantic_scholar",
        quote="Retrieval augmented generation significantly reduces hallucination rates",
        found_by="paper_agent",
        task="find RAG papers",
    )
    return store


# --- Add and basic retrieval ---

def test_add_returns_id():
    store = EvidenceStore()
    eid = store.add("claim", "title", "http://url", "arxiv", "quote", "paper_agent", "task")
    assert eid == "E1"
    assert len(store) == 1


def test_add_sequential_ids():
    store = EvidenceStore()
    e1 = store.add("claim 1", "t", "http://a", "arxiv", "q1", "paper_agent", "task")
    e2 = store.add("claim 2", "t", "http://b", "arxiv", "q2", "paper_agent", "task")
    assert e1 == "E1"
    assert e2 == "E2"
    assert len(store) == 2


def test_get_by_id():
    store = _make_store_with_items()
    item = store.get("E1")
    assert item is not None
    assert item.claim == "LLMs hallucinate facts"


def test_get_nonexistent_returns_none():
    store = _make_store_with_items()
    assert store.get("E999") is None


# --- Deduplication ---

def test_duplicate_claim_same_source_rejected():
    store = EvidenceStore()
    e1 = store.add("LLMs hallucinate", "Paper A", "http://a", "arxiv", "q", "paper_agent", "t")
    e2 = store.add("LLMs hallucinate", "Paper A", "http://a", "arxiv", "q", "paper_agent", "t")
    assert e1 == "E1"
    assert e2 is None
    assert len(store) == 1


def test_duplicate_claim_case_insensitive():
    store = EvidenceStore()
    store.add("LLMs Hallucinate Facts", "Paper A", "http://a", "arxiv", "q", "paper_agent", "t")
    dup = store.add("llms hallucinate facts", "Paper A", "http://a", "arxiv", "q", "paper_agent", "t")
    assert dup is None
    assert len(store) == 1


def test_same_claim_different_source_allowed():
    store = EvidenceStore()
    e1 = store.add("LLMs hallucinate", "Paper A", "http://a", "arxiv", "q", "paper_agent", "t")
    e2 = store.add("LLMs hallucinate", "Paper B", "http://b", "arxiv", "q", "paper_agent", "t")
    assert e1 == "E1"
    assert e2 == "E2"
    assert len(store) == 2


# --- Status transitions ---

def test_initial_status_unverified():
    store = _make_store_with_items()
    assert all(e.status == "unverified" for e in store.all())
    assert len(store.unverified()) == 2
    assert len(store.supported()) == 0


def test_set_status_supported():
    store = _make_store_with_items()
    ok = store.set_status("E1", "supported", "quote matches claim")
    assert ok is True
    assert store.get("E1").status == "supported"
    assert store.get("E1").checker_note == "quote matches claim"
    assert len(store.supported()) == 1
    assert len(store.unverified()) == 1


def test_set_status_unsupported():
    store = _make_store_with_items()
    store.set_status("E1", "unsupported", "quote not found in source")
    assert store.get("E1").status == "unsupported"


def test_set_status_contradicted():
    store = _make_store_with_items()
    store.set_status("E1", "contradicted", "source says opposite")
    assert store.get("E1").status == "contradicted"


def test_set_status_invalid_raises():
    store = _make_store_with_items()
    try:
        store.set_status("E1", "maybe")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_set_status_nonexistent_returns_false():
    store = _make_store_with_items()
    assert store.set_status("E999", "supported") is False


def test_by_status():
    store = _make_store_with_items()
    store.set_status("E1", "supported")
    store.set_status("E2", "unsupported")
    assert len(store.by_status("supported")) == 1
    assert len(store.by_status("unsupported")) == 1
    assert len(store.by_status("unverified")) == 0


# --- Serialisation ---

def test_to_dict_and_from_dict_roundtrip():
    store = _make_store_with_items()
    store.set_status("E1", "supported", "verified")

    data = store.to_dict()
    restored = EvidenceStore.from_dict(data)

    assert len(restored) == 2
    assert restored.get("E1").status == "supported"
    assert restored.get("E1").checker_note == "verified"
    assert restored.get("E2").status == "unverified"
    assert restored.get("E2").claim == "RAG reduces hallucination"


def test_to_dict_preserves_next_id():
    store = _make_store_with_items()
    data = store.to_dict()
    restored = EvidenceStore.from_dict(data)
    e3 = restored.add("new claim", "Paper C", "http://c", "web", "q", "web_agent", "t")
    assert e3 == "E3"


# --- Summary ---

def test_summary_for_supervisor_contains_counts():
    store = _make_store_with_items()
    store.set_status("E1", "supported")
    summary = store.summary_for_supervisor()
    assert "2 total" in summary
    assert "1 supported" in summary
    assert "1 unverified" in summary
    assert "[E1]" in summary
    assert "[E2]" in summary


def test_summary_no_raw_source_text():
    store = EvidenceStore()
    store.add(
        claim="test claim",
        source_title="Paper",
        source_url="http://paper",
        source_type="arxiv",
        quote="This is a long verbatim quote from the paper that should not appear in the summary",
        found_by="paper_agent",
        task="task",
    )
    summary = store.summary_for_supervisor()
    assert "long verbatim quote" not in summary
