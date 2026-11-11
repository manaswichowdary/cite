from types import SimpleNamespace as NS

from cite.checker import (JudgeResult, OpenAIJudge, Part, best_passages, check_answer,
                          quote_in_page, split_claims)
from cite.search import SearchResult

PAGE_A = ("DuckDB runs analytical SQL queries directly on local files such as CSV and Parquet.\n"
          "It does not need a separate server process to work.")
PAGE_B = "Our company was founded in 2010 and sells garden furniture to customers worldwide."
CLAIM = "DuckDB runs analytical SQL queries on local files without a separate server process [1]."
Q1 = "runs analytical SQL queries directly on local files"
Q2 = "does not need a separate server process"


def src(text):
    return SearchResult(title="t", url="https://x", snippet="", page_text=text, score=1.0)


class FakeJudge:
    def __init__(self, results):
        self.results, self.calls = list(results), 0

    def judge(self, claim, passages):
        self.calls += 1
        return self.results.pop(0)


def test_split_claims_handles_bullets_headers_and_citation_placement():
    answer = ("DuckDB runs analytical queries on local files [1][2]. It needs no server. [3]\n"
              "Common uses include:\n"
              "- Powering interactive data applications and dashboards [3].\n"
              "- Fast.")
    got = split_claims(answer)
    assert got[0] == ("DuckDB runs analytical queries on local files [1][2].", [1, 2])
    assert len(got) == 2                                 # "It needs no server" is too short to check
    assert ("Powering interactive data applications and dashboards [3].", [3]) in got
    assert all("Common uses include" not in s for s, _ in got)


def test_supported_only_when_every_part_has_a_real_quote():
    judge = FakeJudge([JudgeResult([Part("queries local files", Q1), Part("no server", Q2)], 0.001)])
    report = check_answer(CLAIM, {1: src(PAGE_A)}, judge)
    c = report.claims[0]
    assert c.status == "supported" and c.source == 1 and c.quotes == [Q1, Q2]
    assert report.support_rate == 1.0 and report.cost_usd == 0.001


def test_one_missing_fact_makes_it_partial():
    judge = FakeJudge([JudgeResult([Part("queries local files", Q1), Part("no server", "")], 0.0)])
    c = check_answer(CLAIM, {1: src(PAGE_A)}, judge).claims[0]
    assert c.status == "partial" and "1 of 2 facts found" in c.note and "no server" in c.note


def test_made_up_quotes_are_rejected():
    judge = FakeJudge([JudgeResult([Part("fast", "DuckDB is the fastest database ever built")], 0.0)])
    c = check_answer(CLAIM, {1: src(PAGE_A)}, judge).claims[0]
    assert c.status == "unsupported" and "no quote the judge gave" in c.note


def test_second_source_can_save_the_claim():
    judge = FakeJudge([JudgeResult([Part("a", "invented words that are not there")]),
                       JudgeResult([Part("a", Q1), Part("b", Q2)])])
    c = check_answer(CLAIM.replace("[1]", "[1][2]"), {1: src(PAGE_A), 2: src(PAGE_A)}, judge).claims[0]
    assert c.status == "supported" and c.source == 2


def test_empty_judge_answer_is_unsupported():
    c = check_answer(CLAIM, {1: src(PAGE_A)}, FakeJudge([JudgeResult([])])).claims[0]
    assert c.status == "unsupported"


def test_bad_citation_and_uncited_sentences():
    judge = FakeJudge([])
    report = check_answer("DuckDB runs analytical SQL queries on local files [9].\n"
                          "DuckDB also handles parquet data very quickly.", {1: src(PAGE_A)}, judge)
    assert [c.status for c in report.claims] == ["bad_citation", "uncited"]
    assert report.cited == 1 and judge.calls == 0


def test_no_matching_passage_means_no_judge_call():
    judge = FakeJudge([])
    report = check_answer(CLAIM, {1: src(PAGE_B)}, judge)
    assert report.claims[0].status == "unsupported" and judge.calls == 0


def test_best_passages_and_quote_matching_ignore_case_and_spacing():
    assert best_passages("DuckDB runs analytical SQL queries on local files", PAGE_A)
    assert quote_in_page("RUNS  analytical SQL queries\ndirectly", PAGE_A)
    assert not quote_in_page("it is", "it is a database")          # filler is not evidence
    assert not quote_in_page("the", "the database")


def reply(content, pt=500, ct=20):
    return NS(choices=[NS(message=NS(content=content))], usage=NS(prompt_tokens=pt, completion_tokens=ct))


class FakeClient:
    def __init__(self, content):
        self.chat = NS(completions=NS(create=lambda **kw: reply(content)))


def test_openai_judge_parses_parts_and_survives_garbage():
    good = OpenAIJudge(FakeClient('{"parts": [{"part": "p", "quote": "abc"}, {"part": "q"}]}'),
                       "gpt-4o-mini").judge("c", ["p"])
    assert [(p.part, p.quote) for p in good.parts] == [("p", "abc"), ("q", "")] and good.cost_usd > 0
    assert OpenAIJudge(FakeClient("not json"), "gpt-4o-mini").judge("c", ["p"]).parts == []
    assert OpenAIJudge(FakeClient('{"parts": "oops"}'), "gpt-4o-mini").judge("c", ["p"]).parts == []


# ---------- added after the first planted error run ----------

def test_abbreviations_and_initials_do_not_split_a_sentence():
    got = split_claims("The protocol was designed by Dr. Anika Rao in 2017 at the institute [1]. "
                       "It was reviewed by J. Smith and later published [2].")
    assert [c for _, c in got] == [[1], [2]]
    assert got[0][0].startswith("The protocol was designed by Dr. Anika Rao")


def test_punctuation_differences_do_not_break_a_quote():
    page = "Its founder, Joseph Okafor, started the company in a rented kitchen in 1989."
    assert quote_in_page("founder Joseph Okafor started the company", page)
    assert not quote_in_page("founder Joseph Mwangi started the company", page)
    assert not quote_in_page("90 millimetres of rain", "less than 900 millimetres of rain")


def one(sentence, quotes, page):
    judge = FakeJudge([JudgeResult([Part("p", q) for q in quotes], 0.0)])
    return check_answer(sentence, {1: src(page)}, judge).claims[0]


def test_a_real_quote_with_a_different_number_is_only_partial():
    page = "The bridge opened in 1931 and spans 640 metres across the Oska River."
    c = one("The bridge opened in 1931 and spans 460 metres across the Oska River [1].",
            ["The bridge opened in 1931", "spans 640 metres across the Oska River"], page)
    assert c.status == "partial" and "460" in c.note
    ok = one("The bridge opened in 1931 and spans 640 metres across the Oska River [1].",
             ["The bridge opened in 1931", "spans 640 metres across the Oska River"], page)
    assert ok.status == "supported"


def test_a_name_missing_from_the_page_is_caught_but_pronoun_quotes_are_fine():
    page = "The Northline Express runs eight times a day. A one way adult ticket costs 38 euros."
    quote = "A one way adult ticket costs 38 euros"
    bad = one("A one way adult ticket on the Southline Express costs 38 euros [1].", [quote], page)
    assert bad.status == "partial" and "Southline" in bad.note
    good = one("A one way adult ticket on the Northline Express costs 38 euros [1].", [quote], page)
    assert good.status == "supported"


def test_a_claim_with_not_needs_a_quote_with_not():
    page = "Benchmarks show it is about 20 percent slower than quicksort on random data."
    quote = "it is about 20 percent slower than quicksort on random data"
    bad = one("Benchmarks show the sort is not slower than quicksort on random data [1].", [quote], page)
    assert bad.status == "partial" and "'not'" in bad.note
    page2 = "DuckDB does not need a separate server process to work in most cases."
    ok = one("DuckDB needs no separate server process to work in most cases [1].",
             ["does not need a separate server process to work"], page2)
    assert ok.status == "supported"


def test_sentences_are_checked_in_parallel_but_returned_in_order():
    answer = "\n".join(f"- Sentence number {i} says something about the bridge opening [1]." for i in range(6))
    seen = []
    report = check_answer(answer, {1: src("nothing here")}, FakeJudge([]), on_claim=seen.append)
    assert [c.sentence for c in report.claims] == [c.sentence for c in seen]
    assert [c.sentence.split()[2] for c in report.claims] == [str(i) for i in range(6)]


def test_without_and_cannot_are_not_treated_as_contradictions():
    page = "A distributed store can only guarantee two of the three properties during a partition."
    quote = "can only guarantee two of the three properties during a partition"
    c = one("A distributed store cannot guarantee all three properties during a partition [1].", [quote], page)
    assert c.status == "supported"
    page2 = "Containers share the host kernel and run in isolated processes on the same machine."
    c2 = one("Containers run on the same machine without sharing processes [1].",
             ["run in isolated processes on the same machine"], page2)
    assert c2.status == "supported"


def test_us_and_united_states_count_as_the_same_name():
    page = "The Federal Reserve is the central bank of the U.S. and sets the federal funds rate."
    c = one("The Federal Reserve is the central bank of the United States [1].",
            ["The Federal Reserve is the central bank of the U.S."], page)
    assert c.status == "supported"


def test_short_quotes_with_a_real_word_or_number_are_accepted():
    page = "Kubernetes offers self-healing and autoscaling. Its founder, Joseph Okafor, started it in 1989."
    for quote in ("self-healing", "Joseph Okafor", "in 1989", "autoscaling"):
        assert quote_in_page(quote, page), quote
    assert not quote_in_page("self-healing", "Kubernetes offers autoscaling")     # must really be there


def test_short_true_quotes_make_a_supported_claim():
    page = ("Brindle Foods employs about 2,300 people across five factories. "
            "Its founder, Joseph Okafor, started the company in a rented kitchen in 1989.")
    c = one("The founder Joseph Okafor started Brindle Foods in a rented kitchen in 1989 [1].",
            ["founder, Joseph Okafor", "started the company in a rented kitchen", "in 1989"], page)
    assert c.status == "supported", c.note
