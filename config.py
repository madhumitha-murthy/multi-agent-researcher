# config.py - Settings for the multi-agent researcher

import os

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# ArXiv
ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "cs.IR"]
ARXIV_MAX_PER_QUERY = 10

# Semantic Scholar
S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_MAX_PER_QUERY = 10
S2_FIELDS = "title,abstract,url,year,citationCount,authors"

# Agent limits
MAX_COLLECTOR_CALLS = 4  # tool calls per collector
MAX_CRITIC_ROUNDS = 2    # how many times critic can ask for more research
GEMINI_RATE_DELAY = 4.5  # seconds between Gemini calls (free tier: 15/min)
