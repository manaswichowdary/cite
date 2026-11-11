import json
from pathlib import Path

from cite.checker import JudgeResult, Part
from cite.evals import percentile, run_questions, score_planted, summarize
from cite.config import Settings

ROOT = Path(__file__).resolve().parent.parent


class CopyJudge:
    """A judge that only accepts claims that appear word for word in the passage."""

    def judge(self, claim, passages):
        text = " ".join(passages)
        sentence = claim.split(" [")[0].rstrip(".")
        return JudgeResult([Part("claim", sentence if sentence in text else "")], 0.001)


def test_real_case_file_is_well_formed():
    cases = json.loads((ROOT / "evals" / "planted_cases.json").read_text())
    assert len(cases) == 20
    for c in cases:
        for k in ("true", "wrong_number", "wrong_name", "extra_fact", "negation"):
            assert len(c[k].split()) >= 5 and c[k] != c["true" if k != "true" else "passage"]
    questions = json.loads((ROOT / "evals" / "questions.json").read_text())
    assert len(questions) == 30 and [q["id"] for q in questions] == list(range(1, 31))


def test_score_planted_counts_caught_misses_and_false_alarms():
    cases = [{"id": "a1", "passage": "Blorp City has 5000 residents and a large harbour area.",
              "true": "Blorp City has 5000 residents and a large harbour area.",
              "wrong_number": "Blorp City has 9000 residents and a large harbour area.",
              "wrong_name": "Blorp Town has 5000 residents and a large harbour area.",
              "extra_fact": "Blorp City has 5000 residents and a large harbour area.",   # a miss on purpose
              "negation": "Blorp City has no residents and a large harbour area."}]
    r = score_planted(cases, CopyJudge())
    assert r["planted_claims"] == 4 and r["planted_caught"] == 3 and r["catch_rate"] == 0.75
    assert r["true_acceptance_rate"] == 1.0 and r["false_alarms"] == []
    assert [m["kind"] for m in r["misses"]] == ["extra_fact"]
    assert r["catch_rate_by_kind"]["extra_fact"] == {"total": 1, "caught": 0, "rate": 0.0}
    assert r["cost_usd"] > 0


def row(i, supported, cited, uncited=0, cost=0.002, latency=5.0, searches=2, stopped="done"):
    claims = [{"status": "supported"}] * supported + [{"status": "partial"}] * (cited - supported)
    claims += [{"status": "uncited"}] * uncited
    return {"id": i, "question": f"q{i}", "stopped": stopped, "searches": searches, "latency_s": latency,
            "cost_usd": cost, "claims": claims,
            "summary": {"cited": cited, "supported": supported, "uncited": uncited}}


def test_summarize_and_percentile():
    m = summarize([row(1, 4, 4), row(2, 2, 4, uncited=1), {"id": 3, "question": "q3", "error": "X"}])
    assert (m["questions"], m["answered"], m["errors"]) == (3, 2, 1)
    assert m["cited_claims"] == 8 and m["supported_claims"] == 6 and m["support_rate"] == 0.75
    assert m["uncited_sentences"] == 1 and m["answers_with_every_claim_supported"] == 1
    assert m["total_cost_usd"] == 0.004 and m["stopped"] == {"done": 2}
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([], 95) == 0.0


class FakeResult:
    def __init__(self, i):
        self.i = i
        self.cost_usd = 0.01

    def to_dict(self):
        return row(self.i, 1, 1, cost=0.01)


def test_run_questions_resumes_limits_and_stops_on_budget(tmp_path):
    qs = [{"id": i, "question": f"q{i}"} for i in range(1, 6)]
    calls = []

    def run(question, *args):
        calls.append(question)
        return FakeResult(int(question[1:]))

    log = lambda s: None    # noqa: E731
    run_questions(qs, Settings(), None, None, tmp_path, limit=2, run=run, log=log)
    assert calls == ["q1", "q2"]
    run_questions(qs, Settings(), None, None, tmp_path, max_usd=0.015, run=run, log=log)
    assert calls == ["q1", "q2", "q3", "q4"]          # q1 and q2 were skipped; budget stopped it after two
    m = run_questions(qs, Settings(), None, None, tmp_path, run=run, log=log)
    assert calls[-1] == "q5" and m["answered"] == 5


def test_three_failures_in_a_row_stop_the_run(tmp_path):
    qs = [{"id": i, "question": f"q{i}"} for i in range(1, 8)]
    seen = []

    def run(question, *args):
        seen.append(question)
        raise RuntimeError("bad key")

    m = run_questions(qs, Settings(), None, None, tmp_path, run=run, log=lambda s: None)
    assert len(seen) == 3 and m["errors"] == 3


def test_show_prints_only_claims_that_were_not_supported(tmp_path, capsys):
    from cite.evals import main
    rows = [{"id": 1, "question": "q", "claims": [
        {"status": "supported", "sentence": "good one", "note": ""},
        {"status": "partial", "sentence": "weak one", "note": "2 of 3 facts found"}]}]
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert main(["show", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "weak one" in out and "2 of 3 facts found" in out and "good one" not in out


def test_readme_block_is_filled_from_the_result_files(tmp_path):
    from cite.evals import update_readme
    planted = {"catch_rate_by_kind": {"wrong_number": {"total": 20, "caught": 20, "rate": 1.0},
                                      "negation": {"total": 20, "caught": 19, "rate": 0.95}},
               "planted_claims": 40, "planted_caught": 39, "true_claims": 20, "true_accepted": 19}
    metrics = {"questions": 30, "answered": 30, "supported_claims": 119, "cited_claims": 142,
               "answers_with_every_claim_supported": 13, "uncited_sentences": 0, "avg_searches": 1.03,
               "latency_p50_s": 9.74, "latency_p95_s": 11.96, "avg_cost_usd": 0.00125}
    (tmp_path / "p.json").write_text(json.dumps(planted))
    (tmp_path / "m.json").write_text(json.dumps(metrics))
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- RESULTS:START -->\nold text\n<!-- RESULTS:END -->\nafter\n")
    update_readme(readme, tmp_path / "p.json", tmp_path / "m.json")
    update_readme(readme, tmp_path / "p.json", tmp_path / "m.json")      # running twice changes nothing
    text = readme.read_text()
    assert text.startswith("before\n") and text.endswith("after\n") and "old text" not in text
    assert "| Negation | 19 of 20 (95%) |" in text and "119 of 142 (84%)" in text
    assert text.count("<!-- RESULTS:START -->") == 1
