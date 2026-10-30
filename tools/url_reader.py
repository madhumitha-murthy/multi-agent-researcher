# tools/url_reader.py - Fetch and extract text from a URL
#
# Security note: fetched content is fed into the LLM agent's context.
# A malicious page could contain prompt-injection text. We strip HTML tags
# and truncate, but do not attempt full injection detection. Treat agent
# outputs as untrusted when using URLs from the open web.

import ipaddress
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

TOOL_SCHEMA = {
    "name": "read_url",
    "description": (
        "Fetch a web page and extract its text content. "
        "Use this to read the full content of an article or blog post found by search_web. "
        "Only URLs that appeared in a previous search result can be fetched."
    ),
    "params": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch and read"},
        },
        "required": ["url"],
    },
}

# Private/localhost patterns to block
_PRIVATE_HOSTS = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|10\.\d+\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+|\[::1\])$",
    re.IGNORECASE,
)


def is_url_safe(url: str) -> tuple[bool, str]:
    """Check if a URL is safe to fetch (http(s), not private/localhost)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "invalid URL"

    if parsed.scheme not in ("http", "https"):
        return False, f"blocked scheme: {parsed.scheme}"

    host = parsed.hostname or ""
    if _PRIVATE_HOSTS.match(host):
        return False, f"blocked private/localhost address: {host}"

    # Also check if the host resolves to a private IP
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_reserved:
            return False, f"blocked private IP: {host}"
    except ValueError:
        pass  # hostname, not IP — that's fine

    return True, ""


def read_url(url: str, allowed_urls: set[str] | None = None) -> dict:
    """Fetch a URL and extract text.

    Args:
        url: URL to fetch
        allowed_urls: if provided, only URLs in this set may be fetched.
            This restricts read_url to URLs that appeared in prior search results.
    """
    # URL safety check
    safe, reason = is_url_safe(url)
    if not safe:
        return {"error": f"Refused to fetch {url}: {reason}"}

    # Allowlist check (v2 workers pass this)
    if allowed_urls is not None and url not in allowed_urls:
        return {"error": f"URL not in search results: {url}. Only fetch URLs from previous tool results."}

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


def wrap_untrusted(tool_result: dict, url: str = "") -> str:
    """Wrap tool result text in untrusted_source tags for the agent prompt."""
    content = ""
    if isinstance(tool_result.get("result"), list):
        for item in tool_result["result"]:
            title = item.get("title", "")
            abstract = item.get("abstract", item.get("content", ""))
            item_url = item.get("url", url)
            content += f"<untrusted_source url=\"{item_url}\">\n{title}\n{abstract}\n</untrusted_source>\n"
    elif "error" in tool_result:
        content = f"Error: {tool_result['error']}"
    else:
        content = str(tool_result)
    return content
