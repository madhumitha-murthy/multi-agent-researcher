# eval/run_eval.py - Run evaluation questions and compute metrics (v2)
# Usage: python -m eval.run_eval

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def compute_metrics(records: list[dict]) -> dict:
    """Compute aggregate metrics from v2 evaluation records."""
    completed = [r for r in records if not r.get("error")]
    with_claims = [r for r in completed if r["supported_claims"] > 0]

    return {
        "questions": len(records),
        "completed": len(completed),
        "errors": len(records) - len(completed),
        "avg_supported_claims": round(mean(r["supported_claims"] for r in completed), 1) if completed else 0,
        "avg_total_claims": round(mean(r["total_claims"] for r in completed), 1) if completed else 0,
        "claims_found_rate": round(len(with_claims) / len(completed), 3) if completed else 0,
        "avg_supervisor_turns": round(mean(r["supervisor_turns"] for r in completed), 1) if completed else 0,
        "avg_latency_s": round(mean(r["latency_s"] for r in completed), 1) if completed else 0,
        "avg_report_length": round(mean(r["report_length"] for r in completed), 0) if completed else 0,
        "avg_cost_usd": round(mean(r["cost_usd"] for r in completed), 4) if completed else 0,
        "avg_citation_coverage": round(mean(
            r["citation_coverage"] for r in completed if r.get("citation_coverage") is not None
        ), 3) if any(r.get("citation_coverage") is not None for r in completed) else 0,
        "finish_reasons": _count_values(completed, "stop_reason"),
    }


def _count_values(records: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for r in records:
        val = r.get(key, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts


def main():
    load_dotenv()
    from graph import build_graph, run_research
    from llm import GeminiModel

    questions = json.loads(QUESTIONS_FILE.read_text())
    model = GeminiModel()
    graph = build_graph(model)
    records = []

    for i, q in enumerate(questions, 1):
        print(f"[eval] {i}/{len(questions)} {q['id']}: {q['question'][:60]}...")
        record = {
            **q, "supported_claims": 0, "total_claims": 0,
            "supervisor_turns": 0, "latency_s": 0,
            "report_length": 0, "stop_reason": "",
            "cost_usd": 0.0, "citation_coverage": None,
            "error": None,
        }

        try:
            result = run_research(graph, q["question"])
            budget = result.get("budget", {})

            # Extract citation_coverage from the writer trace entry
            citation_cov = None
            for entry in reversed(result.get("trace", [])):
                if entry.get("agent") == "writer" and "citation_coverage" in entry:
                    citation_cov = entry["citation_coverage"]
                    break

            record.update({
                "supported_claims": result["supported_claims"],
                "total_claims": result["total_claims"],
                "supervisor_turns": result["supervisor_turns"],
                "latency_s": result["latency_s"],
                "report_length": len(result["report"]),
                "stop_reason": result["stop_reason"],
                "cost_usd": budget.get("cost_usd", 0),
                "citation_coverage": citation_cov,
            })
        except Exception as e:
            record["error"] = str(e)[:200]
            print(f"  error: {record['error']}")

        records.append(record)
        print(
            f"  -> {record['supported_claims']}/{record['total_claims']} claims, "
            f"{record['supervisor_turns']} turns, {record['latency_s']}s, "
            f"${record['cost_usd']:.4f}"
        )
        time.sleep(2)

    metrics = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        **compute_metrics(records),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS_DIR / "eval_records.json").write_text(json.dumps(records, indent=2, default=str))
    print("\n" + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
