# main.py - CLI entry point for the supervisor-based multi-agent researcher
# Usage: python main.py "your research question here"

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from evidence import EvidenceStore
from graph import build_graph, run_research
from llm import GeminiModel

RESULTS_DIR = Path("results")


def save_result(result: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"research_{ts}.json"
    with path.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    return path


def main():
    load_dotenv()
    if len(sys.argv) < 2:
        print('Usage: python main.py "your research question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nResearching: {question}\n")

    model = GeminiModel()
    graph = build_graph(model)
    result = run_research(graph, question)

    # Print trace summary
    print("=" * 60)
    print("AGENT TRACE")
    print("=" * 60)
    for event in result["trace"]:
        agent = event.get("agent", "?")
        if agent == "supervisor":
            print(f"  [Supervisor] → {event.get('decision', '?')} | {event.get('reason', '')[:80]}")
        elif agent in ("paper_agent", "web_agent"):
            print(f"  [{agent}] {event.get('summary', '')}")
        elif agent == "fact_checker":
            print(f"  [Fact-checker] {event.get('summary', '')}")
        elif agent == "writer":
            print(f"  [Writer] {event.get('summary', '')}")

    # Print report
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(result["report"])

    # Print evidence summary
    store = EvidenceStore.from_dict(result["evidence"])
    print("\n" + "=" * 60)
    print(f"EVIDENCE ({len(store)} total, {len(store.supported())} supported)")
    print("=" * 60)
    for e in store.supported():
        print(f"  [{e.id}] {e.claim[:80]} — {e.source_title[:40]}")

    # Stats
    budget = result.get("budget", {})
    print(f"\nStop reason: {result['stop_reason']}")
    print(f"Supervisor turns: {result['supervisor_turns']}")
    print(f"Latency: {result['latency_s']}s")
    print(f"Tokens: {budget.get('tokens_in', '?')}in / {budget.get('tokens_out', '?')}out")
    print(f"Cost: ${budget.get('cost_usd', 0):.4f}")

    path = save_result(result)
    print(f"Full result saved: {path}")


if __name__ == "__main__":
    main()
