# agents/grounding.py - Verify writer's report is supported by collected sources

import re
from functools import lru_cache
from typing import Callable

SENT_THRESHOLD = 0.35  # lower than tech-news-agent because report sentences are more synthesised
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
