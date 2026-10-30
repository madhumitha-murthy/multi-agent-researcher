# agents/fact_checker.py - Verify evidence claims against their source quotes
#
# Two-stage check per evidence item:
# 1. Deterministic: quote must exist verbatim (normalised) in the source text.
# 2. LLM entailment: given only the quote and claim, is the claim supported?
#
# Replaces the old embedding-similarity grounding.py.

import json
import re

from evidence import Evidence, EvidenceStore
from llm import strip_json_fences

BATCH_SIZE = 10  # claims per LLM call

SYSTEM_PROMPT = """You are a fact-checking judge. For each claim-quote pair, decide whether the quote SUPPORTS the claim.

Rules:
- "supported" means the quote alone logically justifies the claim. Topical similarity is NOT support.
- "unsupported" means the quote does not contain enough information to justify the claim.
- "contradicted" means the quote directly contradicts the claim.
- Be strict: if the claim adds specific numbers, names, or conclusions not in the quote, mark unsupported.

Output ONLY valid JSON — a list of objects:
[{"id": "E1", "verdict": "supported|unsupported|contradicted", "reason": "one sentence"}]"""


def normalise_for_match(text: str) -> str:
    """Collapse whitespace and lowercase for verbatim quote matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def quote_in_source(quote: str, source_text: str) -> bool:
    """Check if quote appears verbatim (whitespace/case normalised) in source text."""
    if not quote or not source_text:
        return False
    return normalise_for_match(quote) in normalise_for_match(source_text)


def run_fact_checker(
    model,
    store: EvidenceStore,
    source_texts: dict[str, str],
) -> dict:
    """Check all unverified evidence in the store.

    Args:
        model: ChatModel (or FakeModel for tests)
        store: EvidenceStore with unverified items to check
        source_texts: mapping of source_url -> full text returned by the tool.
            Used for the deterministic quote check.

    Returns:
        dict with summary stats.
    """
    unverified = store.unverified()
    if not unverified:
        return {"checked": 0, "supported": 0, "unsupported": 0, "contradicted": 0,
                "quote_not_found": 0, "total_tokens_in": 0, "total_tokens_out": 0}

    quote_not_found = 0
    needs_llm_check: list[Evidence] = []

    # Stage 1: Deterministic quote check
    for ev in unverified:
        source_text = source_texts.get(ev.source_url, "")
        if not quote_in_source(ev.quote, source_text):
            store.set_status(ev.id, "unsupported", "quote not found in source")
            ev.compute_confidence()
            quote_not_found += 1
        else:
            needs_llm_check.append(ev)

    # Stage 2: LLM entailment check in batches
    total_tokens_in = 0
    total_tokens_out = 0
    llm_supported = 0
    llm_unsupported = 0
    llm_contradicted = 0

    for i in range(0, len(needs_llm_check), BATCH_SIZE):
        batch = needs_llm_check[i:i + BATCH_SIZE]
        verdicts = _llm_check_batch(model, batch)
        total_tokens_in += verdicts["tokens_in"]
        total_tokens_out += verdicts["tokens_out"]

        for eid, verdict, reason in verdicts["results"]:
            if verdict == "supported":
                store.set_status(eid, "supported", reason)
                llm_supported += 1
            elif verdict == "contradicted":
                store.set_status(eid, "contradicted", reason)
                llm_contradicted += 1
            else:
                store.set_status(eid, "unsupported", reason)
                llm_unsupported += 1
            # Compute confidence after verdict is set
            ev = store.get(eid)
            if ev:
                ev.compute_confidence()

    return {
        "checked": len(unverified),
        "supported": llm_supported,
        "unsupported": llm_unsupported + quote_not_found,
        "contradicted": llm_contradicted,
        "quote_not_found": quote_not_found,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
    }


def _llm_check_batch(model, batch: list[Evidence]) -> dict:
    """Run LLM entailment check on a batch of evidence items."""
    pairs_text = ""
    for ev in batch:
        pairs_text += (
            f"\n[{ev.id}]\n"
            f"Claim: {ev.claim}\n"
            f"Quote: \"{ev.quote}\"\n"
        )

    prompt = f"Check each claim-quote pair:\n{pairs_text}"
    messages = [{"role": "user", "text": prompt}]
    resp = model.call(SYSTEM_PROMPT, messages, json_mode=True)

    results = []
    try:
        parsed = json.loads(strip_json_fences(resp.text))
        if not isinstance(parsed, list):
            parsed = [parsed]

        verdict_map = {item.get("id"): item for item in parsed if isinstance(item, dict)}

        for ev in batch:
            entry = verdict_map.get(ev.id, {})
            verdict = entry.get("verdict", "unsupported")
            if verdict not in ("supported", "unsupported", "contradicted"):
                verdict = "unsupported"
            reason = entry.get("reason", "no reason provided")
            results.append((ev.id, verdict, reason))
    except (json.JSONDecodeError, KeyError, TypeError):
        # If parsing fails, mark all as unsupported
        for ev in batch:
            results.append((ev.id, "unsupported", "LLM response parsing failed"))

    return {
        "results": results,
        "tokens_in": resp.input_tokens,
        "tokens_out": resp.output_tokens,
    }
