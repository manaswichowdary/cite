import json

import numpy as np

from cite.checker import JudgeResult, Part, check_answer
from cite.config import load_settings
from cite.evals import recheck_rows
from cite.retrieval import EmbeddingRetriever, HybridRetriever, WordRetriever, make_retriever
from cite.search import SearchResult

# Meaning groups: words in the same group get the same direction, like a tiny embedding model.
GROUPS = [{"car", "cars", "automobile", "automobiles", "vehicle"},
          {"fast", "quick", "speedy", "rapid"},
          {"battery", "batteries", "charge", "charging"},
          {"garden", "plants", "flowers"}]


class FakeEmbedder:
    model_name = "fake-groups"

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), len(GROUPS)), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(".", " ").replace(",", " ").split():
                for g, words in enumerate(GROUPS):
                    if w in words:
                        out[i, g] += 1
        return out


GARDEN = " ".join(["Our garden has many flowers and plants that bloom in spring."] * 9)   # one long passage
PAGE = (GARDEN + "\n" + "The automobile reached top speed in a rapid run thanks to its new batteries. " * 5
        + "\n" + GARDEN)


def test_embeddings_find_a_paraphrase_that_word_matching_misses(tmp_path):
    claim = "The car was fast because of the battery [1]."
    assert WordRetriever().passages(claim, PAGE) == []          # no shared words
    hits = EmbeddingRetriever(FakeEmbedder(), tmp_path).passages(claim, PAGE)
    assert hits and "automobile" in hits[0]


def test_page_vectors_are_cached_on_disk(tmp_path):
    emb = FakeEmbedder()
    r = EmbeddingRetriever(emb, tmp_path)
    r.passages("The car was fast [1].", PAGE)
    r.passages("The vehicle charging was quick [1].", PAGE)
    assert r.pages_embedded == 1 and len(list(tmp_path.glob("*.npy"))) == 1
    again = EmbeddingRetriever(FakeEmbedder(), tmp_path)        # a new process reuses the file
    again.passages("The car was fast [1].", PAGE)
    assert again.pages_embedded == 0


def test_low_scores_are_dropped(tmp_path):
    assert EmbeddingRetriever(FakeEmbedder(), tmp_path).passages("Taxes rose sharply last year [1].", PAGE) == []


def test_hybrid_keeps_meaning_first_and_adds_word_matches_without_repeats(tmp_path):
    h = HybridRetriever(EmbeddingRetriever(FakeEmbedder(), tmp_path, top=8), WordRetriever(), top=3, word_slots=1)
    got = h.passages("The automobile and the garden plants [1].", PAGE)
    assert len(got) == len(set(got)) and len(got) <= 3


def test_make_retriever_names(tmp_path):
    assert make_retriever("words").name == "words"
    assert make_retriever("embeddings", FakeEmbedder(), tmp_path).name == "embeddings"
    assert make_retriever("hybrid", FakeEmbedder(), tmp_path).name == "hybrid"
    try:
        make_retriever("magic")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown names must fail")


def test_setting_defaults_to_embeddings():
    assert load_settings({}).retriever == "embeddings"
    assert load_settings({"CITE_RETRIEVER": "hybrid"}).retriever == "hybrid"


class QuoteJudge:
    """Supports the claim only if it is shown the passage holding the quote."""

    def __init__(self, quote):
        self.quote, self.seen = quote, []

    def judge(self, claim, passages):
        self.seen.append(passages)
        ok = any(self.quote in p for p in passages)
        return JudgeResult([Part("fact", self.quote if ok else "")], 0.0001)


def test_the_checker_uses_the_retriever_it_is_given(tmp_path):
    page = SearchResult("t", "https://a.test", "s", PAGE, 1.0)
    answer = "- The car was fast because of the battery [1]."
    words = check_answer(answer, {1: page}, QuoteJudge("top speed in a rapid run"))
    assert words.claims[0].status == "unsupported" and words.judge_calls == 0
    emb = check_answer(answer, {1: page}, QuoteJudge("top speed in a rapid run"),
                       retriever=EmbeddingRetriever(FakeEmbedder(), tmp_path))
    assert emb.judge_calls == 1


class CacheSearcher:
    credits_used = 0

    def search(self, q):
        return [SearchResult("t", "https://a.test", "s", PAGE, 1.0)]


def test_recheck_rebuilds_sources_from_the_cache_and_scores_again(tmp_path):
    rows = [{"id": 1, "question": "q", "queries": ["cars"], "answer": "- The car was fast because of the battery [1].",
             "sources": [{"n": 1, "title": "t", "url": "https://a.test"}]},
            {"id": 2, "question": "q2", "error": "Timeout"}]
    rep = recheck_rows(rows, CacheSearcher(), QuoteJudge("top speed in a rapid run"),
                       EmbeddingRetriever(FakeEmbedder(), tmp_path), log=lambda s: None)
    assert rep["answers"] == 1 and rep["cited_claims"] == 1 and rep["sources_missing_from_cache"] == 0
    assert rep["retriever"] == "embeddings"
    json.dumps(rep)


def test_pipeline_falls_back_to_words_without_the_model(monkeypatch):
    import importlib.util
    from cite import pipeline
    monkeypatch.setattr(pipeline, "_retrievers", {})
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert pipeline.get_retriever("embeddings").name == "words"
