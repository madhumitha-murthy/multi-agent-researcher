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
            "report_length": 0, "has_both_sources": False, "error": None,
        }

        try:
            result = run_research(graph, q["question"])
            sources = {p.get("source") for p in result["papers"]}
            record.update({
                "n_papers": len(result["papers"]),
                "critic_rounds": result["critic_rounds"],
                "latency_s": result["latency_s"],
                "report_length": len(result["report"]),
                "has_both_sources": "arxiv" in sources and "semantic_scholar" in sources,
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
