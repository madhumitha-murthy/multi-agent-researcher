# tools/url_reader.py - Fetch and extract text from a URL

import requests
from bs4 import BeautifulSoup

TOOL_SCHEMA = {
    "name": "read_url",
    "description": (
        "Fetch a web page and extract its text content. "
        "Use this to read the full content of an article or blog post found by search_web."
    ),
    "params": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch and read"},
        },
        "required": ["url"],
    },
}


def read_url(url: str) -> dict:
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "ResearchAgent/1.0"})
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Truncate to reasonable length
        text = text[:3000]

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        return {
            "title": title,
            "content": text,
            "url": url,
            "source": "web_page",
        }
    except Exception as e:
        return {"error": f"Failed to read {url}: {e}"}
