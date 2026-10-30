# config.py - Settings for the multi-agent researcher

import os

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

# ArXiv
ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "cs.IR"]
ARXIV_MAX_PER_QUERY = 10

# Semantic Scholar
S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_MAX_PER_QUERY = 10
S2_FIELDS = "title,abstract,url,year,citationCount,authors,paperId"

# Agent limits (v1 legacy)
MAX_COLLECTOR_CALLS = 4  # tool calls per collector
MAX_CRITIC_ROUNDS = 2    # how many times critic can ask for more research
GEMINI_RATE_DELAY = 1  # seconds between Gemini calls (paid tier: 2000/min)

# v2 supervisor architecture
MAX_SUPERVISOR_TURNS = 15      # max supervisor decisions per run
MAX_WORKER_TOOL_CALLS = 5     # tool calls per worker agent
MAX_COST_USD = 0.50            # cost limit per question

# Gemini pricing (per 1M tokens)
# Source: https://ai.google.dev/gemini-api/docs/pricing
# gemini-3.7-flash routes to 3.8 Flash. Introductory pricing through 2026-12-31.
PRICE_PER_1M_INPUT = 0.75     # Gemini 3.8 Flash input
PRICE_PER_1M_OUTPUT = 3.75    # Gemini 3.8 Flash output
