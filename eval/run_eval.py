# eval/run_eval.py - Run evaluation questions and compute metrics
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
    completed = [r for r in records if not r.get("error")]
    with_papers = [r for r in completed if r["n_papers"] > 0]
    multi_source = [r for r in completed if r.get("expect_multi_source")]

    grounded_runs = [r for r in completed if r.get("grounding_score", 0) > 0]

    return {
        "questions": len(records),
        "completed": len(completed),
        "errors": len(records) - len(completed),
        "avg_papers": round(mean(r["n_papers"] for r in completed), 1) if completed else 0,
        "avg_latency_s": round(mean(r["latency_s"] for r in completed), 1) if completed else 0,
        "avg_critic_rounds": round(mean(r["critic_rounds"] for r in completed), 2) if completed else 0,
        "papers_found_rate": round(len(with_papers) / len(completed), 3) if completed else 0,
        "multi_source_rate": round(
            sum(1 for r in multi_source if r["has_both_sources"]) / len(multi_source), 3
        ) if multi_source else 0,
        "avg_report_length": round(mean(r["report_length"] for r in completed), 0) if completed else 0,
        "avg_grounding_score": round(mean(r["grounding_score"] for r in grounded_runs), 3) if grounded_runs else 0,
        "grounded_rate": round(sum(1 for r in completed if r.get("grounded")) / len(completed), 3) if completed else 0,
        "total_tool_errors": sum(r.get("tool_errors", 0) for r in completed),
        "avg_total_tokens": round(mean(
            r.get("token_usage", {}).get("total", 0) for r in completed
        ), 0) if completed else 0,
    }


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
            **q, "n_papers": 0, "critic_rounds": 0, "latency_s": 0,
            "report_length": 0, "has_both_sources": False,
            "grounding_score": 0.0, "grounded": False,
            "tool_errors": 0, "token_usage": {},
            "error": None,
        }

        try:
            result = run_research(graph, q["question"])
            sources = {p.get("source") for p in result["papers"]}
            grounding = result.get("grounding", {})
            record.update({
                "n_papers": len(result["papers"]),
                "critic_rounds": result["critic_rounds"],
                "latency_s": result["latency_s"],
                "report_length": len(result["report"]),
                "has_both_sources": "arxiv" in sources and "semantic_scholar" in sources,
                "grounding_score": grounding.get("supported_fraction", 0.0),
                "grounded": grounding.get("grounded", False),
                "tool_errors": result.get("tool_errors", 0),
                "token_usage": result.get("token_usage", {}),
            })
        except Exception as e:
            record["error"] = str(e)[:200]
            print(f"  error: {record['error']}")

        records.append(record)
        print(f"  -> {record['n_papers']} papers, {record['critic_rounds']} critic rounds, {record['latency_s']}s")
        time.sleep(5)

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
