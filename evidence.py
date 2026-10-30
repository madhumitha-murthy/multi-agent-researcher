# evidence.py - Evidence dataclass + EvidenceStore
# The single source of truth for all claims collected during research.

import re
from dataclasses import asdict, dataclass
from typing import Optional

# Semantic dedup threshold — claims with cosine similarity above this are duplicates
SEMANTIC_DEDUP_THRESHOLD = 0.85


@dataclass
class Evidence:
    id: str                 # "E1", "E2", ... assigned by the store
    claim: str              # one atomic factual claim, one sentence
    source_title: str
    source_url: str
    source_type: str        # "arxiv" | "semantic_scholar" | "web" | "github" | "web_page"
    quote: str              # verbatim span copied from the source text
    found_by: str           # "paper_agent" | "web_agent"
    task: str               # the task the worker was given
    status: str = "unverified"   # "unverified" | "supported" | "unsupported" | "contradicted"
    checker_note: str = ""
    confidence: float = 0.0  # 0.0–1.0, computed after fact-checking

    def compute_confidence(self) -> float:
        """Compute confidence score based on status, quote length, and source type.

        Factors:
        - Status: supported=base 0.6, unsupported/contradicted=0.0, unverified=0.1
        - Quote length: longer quotes are more specific (up to +0.2)
        - Source type: academic sources get a bonus (up to +0.2)
        """
        if self.status in ("unsupported", "contradicted"):
            self.confidence = 0.0
            return 0.0
        if self.status == "unverified":
            self.confidence = 0.1
            return 0.1

        # Base score for supported evidence
        score = 0.6

        # Quote length bonus: 0–0.2 scaled by quote length (50–200 chars range)
        qlen = len(self.quote)
        quote_bonus = min(0.2, max(0.0, (qlen - 50) / 750))
        score += quote_bonus

        # Source type bonus
        source_bonus = {
            "arxiv": 0.2,
            "semantic_scholar": 0.2,
            "github": 0.1,
            "web": 0.05,
            "web_page": 0.05,
        }.get(self.source_type, 0.0)
        score += source_bonus

        self.confidence = round(min(1.0, score), 3)
        return self.confidence


class EvidenceStore:
    """Thread-safe evidence store. Workers add evidence; fact-checker updates status."""

    def __init__(self):
        self._items: list[Evidence] = []
        self._next_id: int = 1

    def add(self, claim: str, source_title: str, source_url: str,
            source_type: str, quote: str, found_by: str, task: str,
            semantic_dedup: bool = True) -> Optional[str]:
        """Add evidence. Returns assigned ID, or None if duplicate.

        Dedup is two-tier:
        1. Exact match: same normalised claim text + same source URL → reject
        2. Semantic match: cosine similarity > 0.85 with any existing claim → reject
        """
        norm_claim = _normalise(claim)
        norm_url = source_url.strip()

        for existing in self._items:
            # Tier 1: exact claim + same source
            if _normalise(existing.claim) == norm_claim and existing.source_url.strip() == norm_url:
                return None

        # Tier 2: semantic similarity across all existing claims
        if semantic_dedup and self._items:
            for existing in self._items:
                sim = claim_similarity(claim, existing.claim)
                if sim >= SEMANTIC_DEDUP_THRESHOLD:
                    return None

        eid = f"E{self._next_id}"
        self._next_id += 1
        item = Evidence(
            id=eid,
            claim=claim.strip(),
            source_title=source_title.strip(),
            source_url=source_url.strip(),
            source_type=source_type.strip(),
            quote=quote.strip(),
            found_by=found_by,
            task=task,
        )
        self._items.append(item)
        return eid

    def get(self, eid: str) -> Optional[Evidence]:
        for item in self._items:
            if item.id == eid:
                return item
        return None

    def all(self) -> list[Evidence]:
        return list(self._items)

    def unverified(self) -> list[Evidence]:
        return [e for e in self._items if e.status == "unverified"]

    def supported(self) -> list[Evidence]:
        return [e for e in self._items if e.status == "supported"]

    def by_status(self, status: str) -> list[Evidence]:
        return [e for e in self._items if e.status == status]

    def set_status(self, eid: str, status: str, note: str = "") -> bool:
        """Update status of an evidence item. Returns True if found."""
        valid = {"unverified", "supported", "unsupported", "contradicted"}
        if status not in valid:
            raise ValueError(f"Invalid status {status!r}, must be one of {valid}")
        item = self.get(eid)
        if item is None:
            return False
        item.status = status
        item.checker_note = note
        return True

    def summary_for_supervisor(self) -> str:
        """Compact text for the supervisor: counts + claim one-liners. No raw source text."""
        counts = {}
        for e in self._items:
            counts[e.status] = counts.get(e.status, 0) + 1

        lines = [f"Evidence: {len(self._items)} total — " +
                 ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))]

        for e in self._items:
            lines.append(f"  [{e.id}] ({e.status}) {e.claim[:100]} — src: {e.source_title[:60]}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "next_id": self._next_id,
            "items": [asdict(e) for e in self._items],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceStore":
        store = cls()
        store._next_id = data.get("next_id", 1)
        for item_dict in data.get("items", []):
            store._items.append(Evidence(**item_dict))
        return store

    def __len__(self) -> int:
        return len(self._items)


def _normalise(text: str) -> str:
    """Normalise for deduplication: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


# --- Semantic similarity for dedup ---

_embedding_model = None


def _get_embedding_model():
    """Lazy-load sentence-transformers model (cached after first call)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            return None
    return _embedding_model


def claim_similarity(claim_a: str, claim_b: str) -> float:
    """Compute cosine similarity between two claim strings.

    Returns 0.0 if sentence-transformers is not available.
    """
    model = _get_embedding_model()
    if model is None:
        return 0.0
    embeddings = model.encode([claim_a, claim_b], normalize_embeddings=True)
    return float(embeddings[0] @ embeddings[1])
