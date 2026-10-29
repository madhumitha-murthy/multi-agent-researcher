# Multi-Agent Researcher

A multi-agent system that autonomously researches any topic by coordinating specialized AI agents. Built with LangGraph and Gemini.

```
Question → Planner → Collectors (parallel) → Critic ←→ Gap Filler → Writer → Grounding Check → Report
```

## How It Works

You ask a research question. Four agents collaborate to produce a grounded, cited report:

1. **Planner** decomposes the question into 2-3 focused sub-tasks
2. **Collectors** run in parallel — each is a ReAct agent that autonomously decides which tools to call (ArXiv, Semantic Scholar, web search, GitHub, URL reader)
3. **Critic** evaluates the collected papers, removes irrelevant ones, and identifies gaps. If gaps exist, it sends new sub-tasks back to the collectors (up to 2 feedback rounds)
4. **Writer** synthesises everything into a structured report with citations
5. **Grounding Check** verifies the report is actually supported by the collected sources using embedding similarity

```
                    ┌─────────────────────────────────────────┐
                    │            FEEDBACK LOOP                │
                    │                                         │
  ┌─────────┐   ┌──▼───────┐   ┌────────┐   ┌──────────┐   │
  │ Planner ├──►│Collectors├──►│ Critic ├──►│Gap Filler├───┘
  └─────────┘   │(parallel)│   └───┬────┘   └──────────┘
                └──────────┘       │ approve
                                   ▼
                             ┌──────────┐   ┌───────────┐
                             │  Writer  ├──►│ Grounding │──► Report
                             └──────────┘   └───────────┘
```

## What Makes It Agentic

- **Collectors use the ReAct pattern**: the LLM decides which tools to call, inspects results, and decides what to search next — it's not a fixed pipeline
- **Critic feedback loop**: the critic can reject results and send the collectors back to fill gaps, creating an adaptive research cycle
- **Parallel execution**: collectors run concurrently via ThreadPoolExecutor
- **Tool autonomy**: each collector has 5 tools and chooses which to use based on the sub-task
- **Grounding verification**: the system checks its own output for factual support

## Tools

| Tool | Source | What It Finds |
|------|--------|---------------|
| `search_arxiv` | ArXiv API | Recent AI/ML/NLP preprints |
| `search_semantic_scholar` | Semantic Scholar API | Published papers with citation counts |
| `search_web` | DuckDuckGo | Blog posts, news, industry reports |
| `search_github` | GitHub REST API | Open-source repos and implementations |
| `read_url` | BeautifulSoup | Full text from any URL |

No API keys needed for any tool. Only Gemini requires an API key.

## Setup

```bash
git clone https://github.com/your-username/multi-agent-researcher.git
cd multi-agent-researcher
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

## Usage

```bash
python main.py "What are the latest approaches to reducing hallucinations in LLMs?"
```

Output includes:
- **Agent trace** — which agents ran, how many papers found, critic decisions
- **Research report** — structured with Overview, Key Findings, Approaches Compared, Gaps, References
- **Source list** — all papers with titles, sources, years, citation counts
- **Grounding score** — what fraction of the report is supported by sources
- **JSON result** — full trace saved to `results/`

## Project Structure

```
multi-agent-researcher/
├── main.py                    # CLI entry point
├── graph.py                   # LangGraph orchestrator (state machine + routing)
├── llm.py                     # Gemini wrapper + FakeModel for tests
├── config.py                  # Settings (model, rate limits, tool params)
├── agents/
│   ├── planner.py             # Decomposes question into sub-tasks
│   ├── collector.py           # ReAct agent — searches using tools
│   ├── critic.py              # Evaluates papers, finds gaps
│   ├── writer.py              # Synthesises report from papers
│   └── grounding.py           # Embedding similarity check
├── tools/
│   ├── registry.py            # Central tool registry + executor
│   ├── arxiv_search.py        # ArXiv API search
│   ├── s2_search.py           # Semantic Scholar API search
│   ├── web_search.py          # DuckDuckGo web search
│   ├── github_search.py       # GitHub REST API search
│   └── url_reader.py          # URL content extraction
├── eval/
│   ├── questions.json         # 10 evaluation questions
│   └── run_eval.py            # Batch evaluation with metrics
├── tests/
│   └── test_agents.py         # 17 tests (all agents + graph + grounding)
├── requirements.txt
└── pyproject.toml
```

## Tests

```bash
python -m pytest tests/ -v
```

17 tests cover:
- Each agent individually (planner, collector, critic, writer)
- Grounding check (pass, fail, no sources, sentence splitting)
- Full graph integration (single pass + critic feedback loop)
- Tool registry error handling
- Eval metrics computation

All tests use `FakeModel` — no API calls needed.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Gemini 3.8 Flash (function calling) |
| Orchestration | LangGraph StateGraph |
| Grounding | sentence-transformers (all-MiniLM-L6-v2) |
| Academic Search | ArXiv API, Semantic Scholar API |
| Web Search | DuckDuckGo (no API key) |
| Code Search | GitHub REST API |
| URL Parsing | BeautifulSoup4 |
| Testing | pytest with FakeModel |

## Configuration

Edit `config.py` to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `GEMINI_MODEL` | `gemini-3.8-flash` | Which Gemini model to use |
| `MAX_COLLECTOR_CALLS` | `4` | Tool calls per collector agent |
| `MAX_CRITIC_ROUNDS` | `2` | Max feedback loops before forcing approval |
| `GEMINI_RATE_DELAY` | `4.5s` | Delay between API calls (free tier: 15/min) |

## Evaluation

```bash
python -m eval.run_eval
```

Runs 10 research questions and reports:
- Average papers found per question
- Critic round usage
- Multi-source rate (papers from 2+ sources)
- Latency per question

Results saved to `results/eval_metrics.json`.
