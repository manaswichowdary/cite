from cite.config import load_settings


def test_defaults_when_env_is_empty():
    s = load_settings({})
    assert not s.has_openai and not s.has_tavily
    assert s.model == "gpt-4o-mini"
    assert s.max_usd_per_question == 0.05
    assert s.max_steps == 8


def test_reads_keys_and_overrides():
    s = load_settings({"OPENAI_API_KEY": "a", "TAVILY_API_KEY": "b",
                       "CITE_MODEL": "m", "CITE_MAX_USD_PER_QUESTION": "0.2",
                       "CITE_MAX_STEPS": "3"})
    assert s.has_openai and s.has_tavily
    assert (s.model, s.max_usd_per_question, s.max_steps) == ("m", 0.2, 3)


def test_blank_key_counts_as_missing():
    assert not load_settings({"OPENAI_API_KEY": "   "}).has_openai


def test_keys_never_appear_in_repr():
    s = load_settings({"OPENAI_API_KEY": "secret-one", "TAVILY_API_KEY": "secret-two"})
    assert "secret" not in repr(s)


def test_daily_cap_default_and_override():
    assert load_settings({}).daily_cap_usd == 0.5
    assert load_settings({"CITE_DAILY_CAP_USD": "2"}).daily_cap_usd == 2.0
