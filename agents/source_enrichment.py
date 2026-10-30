# agents/source_enrichment.py - Fetch full text for top search results
#
# After the ReAct search loop, workers call enrich_sources() to upgrade
# abstract snippets (~300 chars) into full page text (~3000 chars).
# This makes quote validation and fact-checking far more reliable.

MAX_URLS_TO_FETCH = 5  # fetch the top N unique URLs


def enrich_sources(
    source_texts: dict[str, str],
    valid_urls: set[str],
    execute_fn,
    budget_calls_remaining: int = 5,
) -> dict:
    """Fetch full text for URLs that only have short abstract snippets.

    Args:
        source_texts: existing url -> text map (from search results)
        valid_urls: all URLs seen in tool results
        execute_fn: tool execution function (name, args) -> dict
        budget_calls_remaining: how many read_url calls to allow

    Returns:
        dict with "source_texts" (updated), "urls_fetched", "fetch_errors"
    """
    # Find URLs with only short text (likely just title + abstract)
    urls_to_fetch = []
    for url in valid_urls:
        existing = source_texts.get(url, "")
        if len(existing) < 500:  # short text, worth fetching full page
            urls_to_fetch.append(url)

    # Limit to top N and budget
    urls_to_fetch = urls_to_fetch[:min(MAX_URLS_TO_FETCH, budget_calls_remaining)]

    fetched = 0
    errors = 0
    for url in urls_to_fetch:
        try:
            result = execute_fn("read_url", {"url": url, "allowed_urls": valid_urls})
            if "error" in result:
                errors += 1
                continue
            # Extract the text content from the result
            items = result.get("result", [])
            if items and isinstance(items, list):
                for item in items:
                    content = item.get("content", "")
                    title = item.get("title", "")
                    if content:
                        full_text = f"{title} {content}"
                        # Only upgrade if we got more text
                        if len(full_text) > len(source_texts.get(url, "")):
                            source_texts[url] = full_text
                            fetched += 1
        except Exception:
            errors += 1

    return {
        "source_texts": source_texts,
        "urls_fetched": fetched,
        "fetch_errors": errors,
    }
