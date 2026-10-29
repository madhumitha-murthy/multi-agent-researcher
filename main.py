# main.py - CLI entry point
# Usage: python main.py "your research question here"

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

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
        if agent == "planner":
            print(f"  [Planner] Sub-tasks: {event['sub_tasks']}")
        elif agent == "collectors_batch":
            print(f"  [Collectors] {event['n_collectors']} agents -> {event['total_papers']} papers ({event['latency_s']}s)")
        elif agent == "critic":
            print(f"  [Critic] Round {event['round']}: {event['verdict']} | Kept {event['papers_kept']} papers")
            if event["gaps"]:
                print(f"           Gaps: {event['gaps']}")
        elif agent == "writer":
            print(f"  [Writer] Report: {event['report_length']} chars ({event['latency_s']}s)")

    # Print report
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(result["report"])

    # Print sources
    print("\n" + "=" * 60)
    print(f"PAPERS FOUND ({len(result['papers'])})")
    print("=" * 60)
    for p in result["papers"]:
        cite = f" | Citations: {p['citations']}" if p.get("citations") else ""
        print(f"  - {p['title']} ({p.get('source', '?')}, {p.get('year', '?')}{cite})")

    # Grounding
    g = result.get("grounding", {})
    grounded = g.get("grounded", "?")
    frac = g.get("supported_fraction", "?")
    print(f"\nGrounding: {'PASS' if grounded else 'WARN'} ({frac} supported)")
    print(f"Total latency: {result['latency_s']}s | Critic rounds: {result['critic_rounds']}")

    path = save_result(result)
    print(f"Full result saved: {path}")


if __name__ == "__main__":
    main()
