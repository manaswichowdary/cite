import json

from cite.config import Settings
from cite.pipeline import run_question
from cite.trace import write_trace
from tests.fakes import FakeClient, FakeSearcher, Q1, Q2

ANSWER = "- DuckDB runs analytical SQL queries on local files without a separate server process [1].\n- It is very popular among data analysts."
SETTINGS = Settings(openai_api_key="k", tavily_api_key="t")


def run(parts, events=None):
    client = FakeClient(ANSWER, parts)
    return run_question("What is DuckDB?", SETTINGS, client, FakeSearcher(),
                        None if events is None else events.append)


def test_events_come_in_order_and_result_is_complete():
    events = []
    result = run([{"part": "local files", "quote": Q1}, {"part": "no server", "quote": Q2}], events)
    assert [e["type"] for e in events] == ["search", "sources", "answer", "claim", "claim", "done"]
    assert events[0]["query"] == "duckdb"
    assert events[1]["sources"] == [{"n": 1, "title": "DuckDB docs", "url": "https://duckdb.test/docs"}]
    s = result.summary()
    assert (s["cited"], s["supported"], s["uncited"], s["support_rate"]) == (1, 1, 1, 1.0)
    assert events[-1]["summary"] == s
    assert result.cost_usd > 0


def test_to_dict_is_json_and_leaves_out_page_text():
    result = run([{"part": "local files", "quote": Q1}, {"part": "no server", "quote": ""}])
    text = json.dumps(result.to_dict())
    assert "page_text" not in text and "snippet" not in text
    d = json.loads(text)
    assert d["claims"][0]["status"] == "partial" and d["queries"] == ["duckdb"]
    assert d["sources"][0]["n"] == 1 and d["judge_calls"] == 1


def test_trace_file_is_written_and_appended(tmp_path):
    result = run([{"part": "x", "quote": Q1}])
    path = write_trace(result, tmp_path)
    write_trace(result, tmp_path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["question"] == "What is DuckDB?"
