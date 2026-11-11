import httpx
import pytest

from cite.config import Settings
from cite.search import SearchError, Searcher

GOOD = {"results": [
    {"title": "A", "url": "https://a.test", "content": "snip a", "raw_content": "full a", "score": 0.9},
    {"title": "no url", "url": "", "content": "x"},
    {"title": "B", "url": "https://b.test", "content": "snip b", "raw_content": None, "score": 0.5},
]}


def make(tmp_path, handler):
    s = Settings(tavily_api_key="tvly-test")
    return Searcher(s, cache_dir=tmp_path, transport=httpx.MockTransport(handler))


def test_parses_results_and_drops_ones_without_url(tmp_path):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read()
        return httpx.Response(200, json=GOOD)

    searcher = make(tmp_path, handler)
    results = searcher.search("What is DuckDB?")
    assert [r.url for r in results] == ["https://a.test", "https://b.test"]
    assert results[0].page_text == "full a"
    assert results[1].page_text == "snip b"       # falls back to the snippet
    assert seen["auth"] == "Bearer tvly-test"
    assert b"What is DuckDB?" in seen["body"]
    assert searcher.credits_used == 1


def test_same_query_is_served_from_cache(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=GOOD)

    searcher = make(tmp_path, handler)
    first = searcher.search("Same question")
    second = searcher.search("  same QUESTION ")   # spacing and case do not matter
    assert first == second
    assert len(calls) == 1
    assert searcher.credits_used == 1 and searcher.cache_hits == 1


def test_cache_survives_a_new_searcher(tmp_path):
    make(tmp_path, lambda r: httpx.Response(200, json=GOOD)).search("persisted")

    def fail(request):
        raise AssertionError("should not call the network")

    again = make(tmp_path, fail)
    assert len(again.search("persisted")) == 2
    assert again.credits_used == 0


def test_bad_status_raises_and_is_not_cached(tmp_path):
    searcher = make(tmp_path, lambda r: httpx.Response(429))
    with pytest.raises(SearchError, match="429"):
        searcher.search("q")
    assert list(tmp_path.iterdir()) == []
    assert searcher.credits_used == 0


def test_missing_key_and_empty_query(tmp_path):
    with pytest.raises(SearchError):
        Searcher(Settings(), cache_dir=tmp_path)
    with pytest.raises(SearchError):
        make(tmp_path, lambda r: httpx.Response(200, json=GOOD)).search("   ")
