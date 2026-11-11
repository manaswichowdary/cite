import json
from types import SimpleNamespace as NS

from cite.agent import Agent, cost_of
from cite.config import Settings
from cite.search import SearchError, SearchResult


def call(query_json, cid="c1"):
    return NS(id=cid, function=NS(name="search_web", arguments=query_json))


def reply(content=None, calls=None, pt=1000, ct=100):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))],
              usage=NS(prompt_tokens=pt, completion_tokens=ct))


class FakeClient:
    """Plays back scripted replies. If asked to answer without tools, it answers."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if kw["tool_choice"] == "none" and self.script and self.script[0].choices[0].message.tool_calls:
            return reply("Forced answer [1].")
        return self.script.pop(0)


class FakeSearcher:
    def __init__(self, by_query=None, error=None):
        self.by_query, self.error, self.queries = by_query or {}, error, []

    def search(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.by_query.get(query, [])


def R(url, text="text"):
    return SearchResult(title=f"T {url}", url=url, snippet=text, page_text=text, score=0.5)


def settings(**kw):
    return Settings(openai_api_key="k", tavily_api_key="t", **kw)


def test_search_then_answer_numbers_sources_and_costs():
    searcher = FakeSearcher({"duckdb": [R("https://a"), R("https://b")]})
    client = FakeClient([reply(calls=[call('{"query": "duckdb"}')]),
                         reply("DuckDB is embedded [1][2].")])
    out = Agent(client, searcher, settings()).ask("What is DuckDB?")
    assert out.text == "DuckDB is embedded [1][2]."
    assert out.stopped == "done" and out.steps == 2 and out.searches == 1
    assert {n: r.url for n, r in out.sources.items()} == {1: "https://a", 2: "https://b"}
    assert abs(out.cost_usd - 2 * cost_of("gpt-4o-mini", 1000, 100)) < 1e-9
    tool_msg = [m for m in client.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "[1] T https://a" in tool_msg["content"] and "[2] T https://b" in tool_msg["content"]


def test_same_url_keeps_its_number_across_searches():
    searcher = FakeSearcher({"q1": [R("https://a")], "q2": [R("https://b"), R("https://a")]})
    client = FakeClient([reply(calls=[call('{"query": "q1"}', "c1")]),
                         reply(calls=[call('{"query": "q2"}', "c2")]),
                         reply("Done [1][2].")])
    out = Agent(client, searcher, settings()).ask("q")
    assert {n: r.url for n, r in out.sources.items()} == {1: "https://a", 2: "https://b"}


def test_answers_directly_when_no_search_is_needed():
    out = Agent(FakeClient([reply("Hello.")]), FakeSearcher(), settings()).ask("hi")
    assert out.text == "Hello." and out.searches == 0 and out.stopped == "done"


def test_last_step_forces_an_answer():
    looping = [reply(calls=[call('{"query": "x"}', f"c{i}")]) for i in range(5)]
    client = FakeClient(looping)
    out = Agent(client, FakeSearcher({"x": [R("https://a")]}), settings(max_steps=3)).ask("q")
    assert [c["tool_choice"] for c in client.calls] == ["auto", "auto", "none"]
    assert out.text == "Forced answer [1]." and out.stopped == "step_limit"


def test_stops_when_the_spending_cap_is_reached():
    client = FakeClient([reply(calls=[call('{"query": "x"}')], pt=2_000_000, ct=0),
                         reply("never reached")])
    out = Agent(client, FakeSearcher({"x": [R("https://a")]}), settings(max_usd_per_question=0.05)).ask("q")
    assert out.stopped == "budget" and len(client.calls) == 1
    assert out.text.startswith("Stopped:")


def test_bad_arguments_and_search_errors_go_back_to_the_model():
    client = FakeClient([reply(calls=[call("not json", "c1")]),
                         reply(calls=[call('{"query": "x"}', "c2")]),
                         reply("I could not search.")])
    out = Agent(client, FakeSearcher(error=SearchError("Tavily returned status 429")), settings()).ask("q")
    tool_texts = [m["content"] for m in client.calls[2]["messages"] if m["role"] == "tool"]
    assert "needs JSON arguments" in tool_texts[0]
    assert "Search failed: Tavily returned status 429" in tool_texts[1]
    assert out.text == "I could not search." and out.sources == {}


def test_unknown_model_is_priced_high():
    assert cost_of("some-new-model", 1_000_000, 0) > cost_of("gpt-4o-mini", 1_000_000, 0)
    assert json.dumps(cost_of("gpt-4o-mini", 1_000_000, 1_000_000)) == "0.75"


def test_events_report_searches_sources_and_answer():
    searcher = FakeSearcher({"duckdb": [R("https://a")]})
    client = FakeClient([reply(calls=[call('{"query": "duckdb"}')]), reply("It is a database [1].")])
    events = []
    out = Agent(client, searcher, settings()).ask("q", events.append)
    assert [e["type"] for e in events] == ["search", "sources", "answer"]
    assert events[0]["query"] == "duckdb" and events[1]["sources"][0]["url"] == "https://a"
    assert events[2]["text"] == out.text and out.queries == ["duckdb"]
