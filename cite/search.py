"""Web search through Tavily, with a disk cache so the same query is never paid for twice."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .config import Settings, load_settings

TAVILY_URL = "https://api.tavily.com/search"
MAX_PAGE_CHARS = 20_000  # keep stored page text bounded


class SearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str      # short summary from Tavily
    page_text: str    # longer page text, used later to check citations
    score: float


class Searcher:
    def __init__(self, settings: Settings, cache_dir: str | Path = "data/cache/search",
                 transport: httpx.BaseTransport | None = None):
        if not settings.has_tavily:
            raise SearchError("TAVILY_API_KEY is not set")
        self._key = settings.tavily_api_key
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(transport=transport, timeout=30.0)
        self.credits_used = 0      # real searches paid for in this session
        self.cache_hits = 0

    def _cache_path(self, query: str, max_results: int) -> Path:
        digest = hashlib.sha256(f"{max_results}|{query.strip().lower()}".encode()).hexdigest()
        return self._dir / f"{digest}.json"

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise SearchError("empty query")
        path = self._cache_path(query, max_results)
        if path.exists():
            self.cache_hits += 1
            return [SearchResult(**r) for r in json.loads(path.read_text())]

        try:
            resp = self._http.post(
                TAVILY_URL,
                headers={"Authorization": f"Bearer {self._key}"},
                json={"query": query, "search_depth": "basic", "max_results": max_results,
                      "include_raw_content": "text"},
            )
        except httpx.HTTPError as err:
            raise SearchError(f"could not reach Tavily: {err.__class__.__name__}") from err
        if resp.status_code != 200:
            raise SearchError(f"Tavily returned status {resp.status_code}")

        self.credits_used += 1  # a basic search costs 1 credit
        results = []
        for item in resp.json().get("results", []):
            if not item.get("url"):
                continue
            results.append(SearchResult(
                title=item.get("title") or item["url"],
                url=item["url"],
                snippet=item.get("content") or "",
                page_text=(item.get("raw_content") or item.get("content") or "")[:MAX_PAGE_CHARS],
                score=float(item.get("score") or 0.0),
            ))
        path.write_text(json.dumps([asdict(r) for r in results]))
        return results


def main() -> int:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Usage: python -m cite.search "your question"')
        return 2
    searcher = Searcher(load_settings())
    for i, r in enumerate(searcher.search(query), 1):
        print(f"{i}. {r.title}\n   {r.url}\n   score {r.score:.2f}, page text {len(r.page_text)} chars")
    print(f"Credits used this run: {searcher.credits_used}, cache hits: {searcher.cache_hits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
