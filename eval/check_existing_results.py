# eval/check_existing_results.py
# Run the fact-checker over the two existing results/research_*.json files.
# Extracts claims from the report, maps them to source papers, and checks them.
#
# Usage: python -m eval.check_existing_results

import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence import EvidenceStore
from agents.fact_checker import run_fact_checker, quote_in_source

RESULTS_DIR = Path(__file__).parent.parent / "results"


def extract_claims_from_report(report: str, papers: list[dict]) -> list[dict]:
    """Extract factual sentences from a report and map cited paper titles.

    Returns list of {claim, source_title, source_url, source_type, quote_candidate}
    where quote_candidate is the paper's abstract (best available source text).
    """
    # Build title -> paper lookup
    title_map = {}
    for p in papers:
        title_map[p["title"].lower().strip()] = p

    # Strip references section
    ref_match = re.search(r"##\s*References", report, re.IGNORECASE)
    body = report[:ref_match.start()] if ref_match else report

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+|\n+", body)

    claims = []
    for sent in sentences:
        sent = sent.strip(" -*\t")
        if len(sent) < 30:
            continue
        # Skip headers
        if sent.startswith("#"):
            continue

        # Find cited paper titles in this sentence (in [brackets])
        cited = re.findall(r"\[([^\]]+)\]", sent)
        clean_sent = re.sub(r"\[.*?\]", "", sent).strip(" ,.")

        if not clean_sent or len(clean_sent) < 25:
            continue

        # Map citations to papers
        for cite_text in cited:
            cite_lower = cite_text.lower().strip()
            # Try exact match
            paper = title_map.get(cite_lower)
            if not paper:
                # Try substring match
                for title_low, p in title_map.items():
                    if cite_lower in title_low or title_low in cite_lower:
                        paper = p
                        break

            if paper:
                abstract = paper.get("abstract", "")
                claims.append({
                    "claim": clean_sent,
                    "source_title": paper["title"],
                    "source_url": paper.get("url", ""),
                    "source_type": paper.get("source", "unknown"),
                    "quote_candidate": abstract,
                })

        # If no citations found, still record the claim (unattributed)
        if not cited:
            claims.append({
                "claim": clean_sent,
                "source_title": "unattributed",
                "source_url": "",
                "source_type": "none",
                "quote_candidate": "",
            })

    return claims


def check_result_file(filepath: Path) -> dict:
    """Run fact-checker over one result file."""
    data = json.loads(filepath.read_text())
    papers = data["papers"]
    report = data["report"]

    claims = extract_claims_from_report(report, papers)

    # Build evidence store
    store = EvidenceStore()
    source_texts = {}

    for c in claims:
        url = c["source_url"]
        if url:
            source_texts[url] = c["quote_candidate"]  # abstract as source text

        store.add(
            claim=c["claim"],
            source_title=c["source_title"],
            source_url=url,
            source_type=c["source_type"],
            quote=c["quote_candidate"][:200] if c["quote_candidate"] else "",
            found_by="v1_report_extraction",
            task="extracted from v1 report",
        )

    # Since v1 didn't store verbatim quotes (only abstracts), we skip the
    # deterministic quote check by putting the abstract as both quote and source.
    # This means we're testing whether the abstract supports the claim.
    result = {
        "file": filepath.name,
        "question": data["question"],
        "total_claims_extracted": len(claims),
        "claims_with_citation": sum(1 for c in claims if c["source_url"]),
        "claims_without_citation": sum(1 for c in claims if not c["source_url"]),
        "evidence_store": store.to_dict(),
        "note": (
            "v1 did not store verbatim quotes. The 'quote' field contains the "
            "abstract, so the deterministic quote check is not meaningful here. "
            "This demonstrates the gap that the new fact-checker addresses."
        ),
    }

    return result


def main():
    result_files = sorted(RESULTS_DIR.glob("research_*.json"))
    if not result_files:
        print("No result files found in results/")
        return

    outputs = []
    for f in result_files:
        print(f"Checking {f.name}...")
        result = check_result_file(f)
        outputs.append(result)
        print(f"  Claims extracted: {result['total_claims_extracted']}")
        print(f"  With citation: {result['claims_with_citation']}")
        print(f"  Without citation: {result['claims_without_citation']}")

    out_path = RESULTS_DIR / "v1_fact_check_analysis.json"
    out_path.write_text(json.dumps(outputs, indent=2, default=str))
    print(f"\nSaved to {out_path}")
    print(
        "\nNote: v1 did not store verbatim quotes from sources. The old grounding "
        "check used embedding similarity (topic overlap), not factual entailment. "
        "The new fact-checker requires verbatim quotes, which v1 cannot provide."
    )


if __name__ == "__main__":
    main()
