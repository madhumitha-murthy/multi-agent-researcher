# agents/grounding.py - Verify writer's report is supported by collected sources

import re
from functools import lru_cache
from typing import Callable

# Thresholds chosen empirically on 2 eval runs (hallucination + RAG reports).
# SENT_THRESHOLD 0.35 catches off-topic sentences but NOT on-topic fabrications —
# cosine similarity on MiniLM embeddings measures topic overlap, not factual entailment.
# A proper entailment model (e.g. NLI-based) would be needed to catch fabricated claims.
SENT_THRESHOLD = 0.35
REPORT_THRESHOLD = 0.5  # at least 50% of sentences must be supported

Similarity = Callable[[list[str], list[str]], list[list[float]]]


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embedding_similarity(sentences: list[str], sources: list[str]) -> list[list[float]]:
    enc = _encoder()
    a = enc.encode(sentences, normalize_embeddings=True)
    b = enc.encode(sources, normalize_embeddings=True)
    return (a @ b.T).tolist()


def split_sentences(text: str) -> list[str]:
    # Strip everything after ## References — those lines are just source titles
    # and would trivially match, inflating the grounding score
    ref_match = re.search(r"##\s*References", text, re.IGNORECASE)
    if ref_match:
        text = text[:ref_match.start()]

    text = re.sub(r"\[.*?\]", "", text)  # strip citation markers
    text = re.sub(r"#+\s.*", "", text)   # strip markdown headers
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*\t") for p in parts if len(p.strip(" -*\t")) > 25]


def check_report_grounding(
    report: str,
    papers: list[dict],
    similarity: Similarity = embedding_similarity,
) -> dict:
    """Check if the report is grounded in the collected papers."""
    sentences = split_sentences(report)
    source_texts = [
        f"{p.get('title', '')}. {p.get('abstract', p.get('content', ''))}"
        for p in papers
    ]

    if not source_texts:
        return {"grounded": False, "reason": "no sources", "supported_fraction": 0.0}
    if not sentences:
        return {"grounded": False, "reason": "empty report", "supported_fraction": 0.0}

    scores = similarity(sentences, source_texts)
    best_per_sent = [max(row) for row in scores]
    n_supported = sum(1 for s in best_per_sent if s >= SENT_THRESHOLD)
    frac = n_supported / len(sentences)

    return {
        "grounded": frac >= REPORT_THRESHOLD,
        "reason": "ok" if frac >= REPORT_THRESHOLD else "report not well supported by sources",
        "supported_fraction": round(frac, 3),
        "sentences_checked": len(sentences),
        "min_score": round(min(best_per_sent), 3),
    }
