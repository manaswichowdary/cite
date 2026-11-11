"""Shared fakes: a model client and a searcher that never touch the network."""

import json
from types import SimpleNamespace as NS

from cite.search import SearchResult

PAGE = ("DuckDB runs analytical SQL queries directly on local files such as CSV and Parquet.\n"
        "It does not need a separate server process to work.")
Q1 = "runs analytical SQL queries directly on local files"
Q2 = "does not need a separate server process"


class FakeSearcher:
    credits_used = 0
    cache_hits = 0

    def search(self, query):
        return [SearchResult("DuckDB docs", "https://duckdb.test/docs", "snippet", PAGE, 0.9)]


def usage(pt=500, ct=50):
    return NS(prompt_tokens=pt, completion_tokens=ct)


class FakeClient:
    """Agent calls: search once, then answer. Judge calls (they ask for JSON): return the given parts."""

    def __init__(self, answer, parts):
        self.answer, self.parts, self.agent_calls = answer, parts, 0
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        if "response_format" in kw:
            body = json.dumps({"parts": self.parts})
            return NS(choices=[NS(message=NS(content=body))], usage=usage(300, 30))
        self.agent_calls += 1
        if self.agent_calls == 1:
            call = NS(id="c1", function=NS(name="search_web", arguments='{"query": "duckdb"}'))
            return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))], usage=usage())
        return NS(choices=[NS(message=NS(content=self.answer, tool_calls=None))], usage=usage(900, 60))
