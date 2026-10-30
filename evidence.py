# evidence.py - Evidence dataclass + EvidenceStore
# The single source of truth for all claims collected during research.

from dataclasses import dataclass, field, asdict
from typing import Optional
import re


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


class EvidenceStore:
    """Thread-safe evidence store. Workers add evidence; fact-checker updates status."""

    def __init__(self):
        self._items: list[Evidence] = []
        self._next_id: int = 1

    def add(self, claim: str, source_title: str, source_url: str,
            source_type: str, quote: str, found_by: str, task: str) -> Optional[str]:
        """Add evidence. Returns assigned ID, or None if duplicate."""
        # Reject exact-duplicate claims from the same source
        norm_claim = _normalise(claim)
        norm_url = source_url.strip()
        for existing in self._items:
            if _normalise(existing.claim) == norm_claim and existing.source_url.strip() == norm_url:
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
