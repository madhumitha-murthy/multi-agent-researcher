# Multi-Agent Researcher

A supervisor-based multi-agent system that autonomously researches any topic using academic and web sources. Built with LangGraph and Gemini, with a production-ready platform layer: OAuth2/JWT authentication, RBAC, agent registry, MCP server, and Kubernetes-ready deployment.

```
Question → Supervisor ←→ Workers (paper_agent, web_agent, fact_checker, writer) → Report
```

## How It Works

You ask a research question. A **supervisor** coordinates four specialist workers to produce a fact-checked, cited report:

1. **Supervisor** receives the question and decides which worker to call next (and why). Every worker returns to the supervisor — hub-and-spoke, not a pipeline.
2. **Paper Agent** searches ArXiv and Semantic Scholar, extracts claims with verbatim quotes, and validates that each quote actually appears in the source text before adding it to the evidence store.
3. **Web Agent** searches DuckDuckGo, GitHub, and reads URLs (restricted to URLs from prior search results; private IPs blocked). Same quote validation as the paper agent.
4. **Fact Checker** verifies unverified evidence in two stages: (1) deterministic — does the quote appear verbatim in the source? (2) LLM entailment — does the source text actually support the claim?
5. **Writer** produces a structured report citing evidence IDs (`[E1][E3]`). Citations pointing to non-existent evidence are stripped. The References section is built in code, not by the LLM.

The supervisor can call workers in any order, repeat them, or skip them. A budget (turns, cost) enforces termination. Once a report is written, the system finishes.

```
                    ┌──────────────────────────────────────┐
                    │         SUPERVISOR (hub)              │
                    │  Decides next worker each turn        │
                    └──┬────┬────┬────┬────────────────────┘
                       │    │    │    │
              ┌────────┘    │    │    └────────┐
              ▼             ▼    ▼             ▼
        ┌───────────┐ ┌─────────┐ ┌───────────┐ ┌────────┐
        │Paper Agent│ │Web Agent│ │Fact Checker│ │ Writer │
        └───────────┘ └─────────┘ └───────────┘ └────────┘
              │             │           │             │
              └─────────────┴───────────┴─────────────┘
                        all return to supervisor
```

## What Makes It Agentic

- **Supervisor routing**: an LLM decides which worker to call next at every turn — not a fixed pipeline
- **ReAct workers**: paper and web agents use a tool-call loop where the LLM chooses which tools to invoke and how many times
- **Evidence store**: a single source of truth for all claims, with status transitions (unverified → supported/unsupported/contradicted)
- **Full-text enrichment**: after search, workers auto-fetch full page text (up to 3000 chars) for top URLs, upgrading abstract snippets into richer source material
- **Two-stage fact-checking**: deterministic quote match + LLM entailment check
- **Quote validation**: fabricated quotes are rejected before evidence enters the store
- **Semantic dedup**: near-identical claims are caught using sentence-transformer cosine similarity (threshold 0.85), preventing redundant evidence
- **Confidence scoring**: each evidence item gets a confidence score (0–1) based on verification status, quote length, and source type — the writer prioritizes high-confidence evidence
- **Code-level guards**: the writer can't run with zero supported claims; budget exhaustion forces the writer or FINISH; unparseable supervisor responses fall back safely
- **Workers fail safely**: an exception is logged, the run continues

## Tools

| Tool | Source | What It Finds |
|------|--------|---------------|
| `search_arxiv` | ArXiv API | Recent AI/ML/NLP preprints |
| `search_semantic_scholar` | Semantic Scholar API | Published papers with citation counts |
| `get_references` / `get_citations` | Semantic Scholar Graph API | Citation graph traversal |
| `search_web` | DuckDuckGo | Blog posts, news, industry reports |
| `search_github` | GitHub REST API | Open-source repos and implementations |
| `read_url` | BeautifulSoup | Full text from URLs seen in prior search results |

No API keys needed for any tool. Only Gemini requires an API key.

## Setup

```bash
git clone https://github.com/madhumitha-murthy/multi-agent-researcher.git
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
- **Agent trace** — supervisor decisions, worker results, evidence counts
- **Research report** — structured with Overview, Key Findings, Approaches Compared, Gaps, Limitations, and code-built References
- **Evidence summary** — all supported claims with IDs, sources, and quotes
- **Budget stats** — turns used, token counts, cost
- **JSON result** — full trace saved to `results/`

## Project Structure

```
multi-agent-researcher/
├── main.py                        # CLI entry point
├── graph.py                       # LangGraph supervisor orchestrator
├── llm.py                         # Gemini wrapper + FakeModel for tests
├── config.py                      # Settings (model, prices, limits)
├── evidence.py                    # Evidence dataclass + EvidenceStore
├── budget.py                      # Turn/cost/token budget tracking
├── agents/
│   ├── supervisor.py              # Routes to workers via JSON-mode LLM call
│   ├── paper_agent.py             # ArXiv + S2 search with quote validation
│   ├── web_agent.py               # Web + GitHub search with URL allowlist
│   ├── fact_checker.py            # Quote match + LLM entailment verification
│   ├── writer.py                  # Report generation with citation post-check
│   └── react_loop.py              # Shared ReAct loop for worker agents
├── tools/
│   ├── registry.py                # Central tool registry + executor
│   ├── arxiv_search.py            # ArXiv API search
│   ├── s2_search.py               # Semantic Scholar API + citation graph
│   ├── web_search.py              # DuckDuckGo web search
│   ├── github_search.py           # GitHub REST API search
│   └── url_reader.py              # URL fetching with safety checks
├── research_platform/
│   ├── gateway.py                 # FastAPI API gateway (OAuth2, RBAC, rate limiting)
│   ├── auth.py                    # JWT tokens, OAuth2 flows, RBAC, agent identity
│   ├── agent_registry.py          # Agent registry: register, discover, health-check
│   ├── mcp_server.py              # MCP server exposing agents as tools
│   └── logging_config.py          # Structured JSON logging with correlation IDs
├── helm/                          # Kubernetes Helm chart (IaC)
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/                 # Deployment, Service, Ingress, Secrets
├── gitops/
│   └── argocd-application.yaml    # ArgoCD GitOps manifest
├── baselines/pipeline/            # v1 pipeline code (planner→collector→critic→writer)
├── eval/
│   ├── questions.json             # Evaluation questions
│   └── run_eval.py                # Batch evaluation with v2 metrics
├── tests/                         # 154 tests across 8 test files
├── results/                       # Example outputs from real runs
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml       # GitHub Actions CI pipeline
├── requirements.txt
└── pyproject.toml
```

## Platform Layer

The `research_platform/` package adds production-ready infrastructure to the multi-agent system.

### API Gateway (`research_platform/gateway.py`)

FastAPI-based API gateway with OAuth2 authentication, RBAC, and rate limiting.

```bash
# Start the gateway
uvicorn research_platform.gateway:app --port 8080

# Or with Docker
docker compose up gateway
```

**Authentication:**
```bash
# Get a JWT token (OAuth2 password flow)
TOKEN=$(curl -s -X POST http://localhost:8080/auth/token \
  -d "username=researcher&password=researcher" | jq -r .access_token)

# Use the token
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/api/agents
```

**Endpoints:**

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/token` | None | OAuth2 login, returns JWT |
| GET | `/auth/me` | Bearer | Current user info |
| POST | `/api/research` | researcher+ | Start a research task |
| GET | `/api/agents` | viewer+ | List registered agents |
| GET | `/api/agents/{id}` | viewer+ | Get agent details |
| POST | `/api/agents` | admin | Register a new agent |
| DELETE | `/api/agents/{id}` | admin | Deregister an agent |
| POST | `/api/agents/{id}/heartbeat` | any | Agent health heartbeat |
| GET | `/api/agents/discover/{cap}` | viewer+ | Discover agents by capability |
| POST | `/api/agent-identities` | admin | Create agent service account |
| GET | `/api/research/health` | None | Health check |

**RBAC roles:** `admin` (full access), `researcher` (research + view), `viewer` (read-only).

### Agent Registry (`research_platform/agent_registry.py`)

Dynamic agent registry with health checking. Agents register their capabilities and tools, and clients discover them at runtime.

### Agent Identity (`research_platform/auth.py`)

Each agent gets a service account with a signed JWT. Agent tokens include `is_agent: true` and role-scoped permissions, enabling machine-to-machine authentication.

### MCP Server (`research_platform/mcp_server.py`)

[Model Context Protocol](https://modelcontextprotocol.io/) server that exposes research agents as MCP tools. Any MCP-compatible client (Claude Desktop, VS Code, etc.) can invoke the research pipeline.

```bash
# Run MCP server (stdio transport)
python -m research_platform.mcp_server
```

**MCP Tools:**
- `research` — Run a full multi-agent research pipeline
- `search_papers` — Search ArXiv and Semantic Scholar
- `search_web_and_github` — Search the web and GitHub
- `list_agents` — List registered agents
- `agent_health` — Check agent health status

### Structured Logging

JSON-formatted logs with correlation IDs that propagate across async calls, enabling distributed tracing.

### Deployment

**Docker:**
```bash
docker compose up --build
```

**Kubernetes (Helm):**
```bash
helm install researcher ./helm \
  --set secrets.geminiApiKey=$GEMINI_API_KEY \
  --set secrets.jwtSecretKey=$(openssl rand -hex 32)
```

**GitOps (ArgoCD):**
```bash
kubectl apply -f gitops/argocd-application.yaml
```

---

## Tests

```bash
python -m pytest tests/ -v
```

154 tests covering:
- **Supervisor**: routing decisions, budget guards, invalid JSON fallback, FINISH-after-report guard
- **Paper/Web agents**: evidence extraction, quote validation, fabricated quote rejection, URL validation
- **Fact checker**: quote-in-source matching, LLM entailment batching, status transitions
- **Writer**: citation stripping, code-built references, citation coverage, needs-more-evidence threshold
- **Evidence store**: add, semantic dedup, confidence scoring, status transitions, serialisation
- **Source enrichment**: full-text fetching, budget respect, error handling
- **Budget**: turn/cost limits, token tracking, exhaustion detection
- **Graph integration**: full scripted run, worker crash recovery, prompt injection resistance, FINISH after report
- **ReAct loop**: tool call budget, error tracking, URL tracking, untrusted content wrapping
- **Platform — Auth**: JWT creation/validation, password hashing, role-based permissions, agent identity tokens
- **Platform — Registry**: agent registration, deregistration, discovery by capability/tool, heartbeat timeout
- **Platform — Gateway**: OAuth2 login, RBAC enforcement (admin/researcher/viewer), correlation IDs, 401/403 responses
- **Platform — Logging**: correlation ID generation and propagation

All tests use `FakeModel` — no API calls needed.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Gemini 3.7 Flash (function calling + JSON mode) |
| Orchestration | LangGraph StateGraph |
| Academic Search | ArXiv API, Semantic Scholar API |
| Web Search | DuckDuckGo (no API key) |
| Code Search | GitHub REST API |
| URL Parsing | BeautifulSoup4 |
| API Gateway | FastAPI with OAuth2/JWT, RBAC, rate limiting |
| Authentication | OAuth2 password flow, JWT (python-jose), bcrypt |
| Agent Protocol | MCP (Model Context Protocol) server |
| Containerisation | Docker, Docker Compose |
| Infrastructure | Helm chart (Kubernetes), ArgoCD (GitOps) |
| CI/CD | GitHub Actions (test + lint + Docker build) |
| Logging | Structured JSON with correlation IDs |
| Testing | pytest with FakeModel (scripted responses), 154 tests |

## Configuration

Edit `config.py` to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `GEMINI_MODEL` | `gemini-3.7-flash` | Which Gemini model to use |
| `MAX_SUPERVISOR_TURNS` | `15` | Max supervisor decisions per run |
| `MAX_WORKER_TOOL_CALLS` | `5` | Tool calls per worker agent |
| `MAX_COST_USD` | `0.50` | Cost limit per question |
| `GEMINI_RATE_DELAY` | `1s` | Delay between API calls |

## Evaluation

```bash
python -m eval.run_eval
```

Runs evaluation questions and reports:
- Supported/total claims per question
- Supervisor turns used
- Citation coverage
- Cost per question
- Latency

Results saved to `results/eval_metrics.json`.

## Limitations

- **Verification is against page text, not full papers.** Workers auto-fetch full page text (up to 3000 chars) after search, but this is still a web-page extract — not a PDF parse. For ArXiv papers, verification is against the abstract page, not the full paper body.
- **Tool errors are common.** ArXiv and Semantic Scholar rate-limit aggressively on free tiers. The system continues with partial results rather than retrying.
- **Prompt injection is mitigated but not eliminated.** Fetched web content is wrapped in `<untrusted_source>` tags and only the wrapped text reaches the model. `read_url` is restricted to URLs from prior search results, and private IPs are blocked. But a sufficiently adversarial page could still influence the model.
- **No full-text retrieval.** The system searches and reads abstracts/snippets. It does not download or parse PDFs.
- **Cost depends on the model.** With Gemini 3.7 Flash at current introductory pricing, a typical run costs $0.01–0.10 depending on the number of supervisor turns and tool calls.
