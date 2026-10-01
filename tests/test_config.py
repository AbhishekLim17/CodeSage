from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codebase_ai.config import Settings, get_settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("INDEX_DIR")  # the autouse fixture redirects it to a temp dir
    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"
    assert s.embedding_provider == "local"
    assert s.anthropic_api_key is None
    assert s.max_file_bytes == 1_000_000
    assert s.index_dir.name == ".codebase_ai"


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("RETRIEVE_TOP_K", "7")
    s = Settings(_env_file=None)
    assert s.llm_provider == "ollama"
    assert s.retrieve_top_k == 7


def test_empty_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    s = Settings(_env_file=None)
    assert s.anthropic_model == "claude-opus-5-5"
    assert s.anthropic_api_key is None


def test_dotenv_file_is_read(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=openai\nOPENAI_MODEL=my-model\nUNKNOWN_SETTING=1\n", encoding="utf-8")
    s = Settings(_env_file=env_file)
    assert s.llm_provider == "openai"
    assert s.openai_model == "my-model"


def test_dotenv_in_working_directory_is_picked_up_by_default(tmp_path):
    (tmp_path / ".env").write_text("EMBEDDING_PROVIDER=ollama\n", encoding="utf-8")
    assert Settings().embedding_provider == "ollama"


def test_environment_beats_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert Settings(_env_file=env_file).llm_provider == "ollama"


def test_invalid_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "skynet")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_secrets_are_not_shown_in_repr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    s = Settings(_env_file=None)
    assert "sk-test-not-a-real-key" not in repr(s)
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-test-not-a-real-key"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("local", "local:BAAI/bge-small-en-v1.5"),
        ("openai", "openai:text-embedding-3-small"),
        ("ollama", "ollama:nomic-embed-text"),
    ],
)
def test_embedding_model_id(provider, expected, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", provider)
    assert Settings(_env_file=None).embedding_model_id() == expected


def test_every_key_in_env_example_is_a_real_setting():
    example = Path(__file__).parent.parent / ".env.example"
    keys = [
        line.split("=", 1)[0].strip()
        for line in example.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    ]
    assert keys, "no settings found in .env.example"
    assert [k for k in keys if k.lower() not in Settings.model_fields] == []


def test_env_example_loads_and_matches_the_defaults():
    example = Path(__file__).parent.parent / ".env.example"
    loaded = Settings(_env_file=example)
    assert loaded == Settings(_env_file=None).model_copy(update={"index_dir": loaded.index_dir})


def test_retrieval_defaults_and_validation(monkeypatch):
    s = Settings(_env_file=None)
    assert (s.retrieval_mode, s.keyword_weight, s.test_penalty) == ("vector", 1.0, 0.5)
    for name, bad in (("RETRIEVAL_MODE", "fuzzy"), ("TEST_PENALTY", "0"), ("TEST_PENALTY", "1.5"), ("KEYWORD_WEIGHT", "-1")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
        monkeypatch.delenv(name)


def test_the_retriever_library_defaults_match_the_settings_defaults():
    import inspect

    from codebase_ai.retrieval.retriever import Retriever

    defaults = {n: p.default for n, p in inspect.signature(Retriever.__init__).parameters.items()}
    s = Settings(_env_file=None)
    assert defaults["mode"] == s.retrieval_mode
    assert defaults["test_penalty"] == s.test_penalty
    assert defaults["keyword_weight"] == s.keyword_weight
    assert defaults["top_k"] == s.retrieve_top_k
    assert defaults["budget_tokens"] == s.context_token_budget


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
