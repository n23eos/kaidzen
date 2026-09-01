"""Сборка бэкендов из конфига и разрешение ключей: всё падает на старте."""
import pytest

from kaidzen.backends.anthropic_api import AnthropicApiBackend
from kaidzen.backends.base import BackendError
from kaidzen.backends.claude_agent import ClaudeAgentBackend
from kaidzen.backends.openai_compat import OpenAICompatBackend
from kaidzen.backends.registry import (build_backends, read_env_file,
                                       resolve_api_key)

SUBSCRIPTION = {"subscription": {"type": "claude_agent_sdk"}}
DEEPSEEK = {"deepseek": {"type": "openai_compat",
                         "base_url": "https://api.deepseek.com",
                         "api_key_env": "DEEPSEEK_API_KEY",
                         "structured_mode": "json_object"}}


def test_subscription_backend_needs_no_key(tmp_path):
    backends = build_backends({"backends": SUBSCRIPTION}, tmp_path)
    assert isinstance(backends["subscription"], ClaudeAgentBackend)


def test_key_taken_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    backends = build_backends({"backends": DEEPSEEK}, tmp_path)
    assert isinstance(backends["deepseek"], OpenAICompatBackend)


def test_key_falls_back_to_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".env").write_text('DEEPSEEK_API_KEY="from-file"\n')
    assert resolve_api_key("DEEPSEEK_API_KEY", tmp_path) == "from-file"


def test_environment_wins_over_env_file(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=from-file\n")
    assert resolve_api_key("DEEPSEEK_API_KEY", tmp_path) == "from-env"


def test_missing_key_fails_at_startup(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(BackendError, match="DEEPSEEK_API_KEY"):
        build_backends({"backends": DEEPSEEK}, tmp_path)


def test_empty_key_fails_at_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "   ")
    with pytest.raises(BackendError, match="DEEPSEEK_API_KEY"):
        build_backends({"backends": DEEPSEEK}, tmp_path)


def test_anthropic_backend_built_with_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    config = {"backends": {"api": {"type": "anthropic",
                                   "api_key_env": "ANTHROPIC_API_KEY"}}}
    backend = build_backends(config, tmp_path)["api"]
    assert isinstance(backend, AnthropicApiBackend)
    assert "sk-ant-secret" not in repr(backend)


def test_unknown_type_rejected(tmp_path):
    with pytest.raises(BackendError, match="type"):
        build_backends({"backends": {"x": {"type": "telepathy"}}}, tmp_path)


def test_missing_api_key_env_rejected(tmp_path):
    config = {"backends": {"x": {"type": "openai_compat"}}}
    with pytest.raises(BackendError, match="api_key_env"):
        build_backends(config, tmp_path)


def test_empty_backends_section_rejected(tmp_path):
    with pytest.raises(BackendError, match="backends"):
        build_backends({"backends": {}}, tmp_path)


def test_env_file_parser_skips_comments_and_junk(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# комментарий\n\nA=1\nмусор без равно\nB='two'\n")
    assert read_env_file(path) == {"A": "1", "B": "two"}


def test_env_file_absent_is_not_an_error(tmp_path):
    assert read_env_file(tmp_path / ".env") == {}


def test_non_dict_backend_spec_rejected(tmp_path):
    with pytest.raises(BackendError, match="словарь"):
        build_backends({"backends": {"x": "subscription"}}, tmp_path)


def test_openai_compat_without_base_url_needs_the_openai_key(tmp_path,
                                                             monkeypatch):
    """Без base_url клиент уходит в api.openai.com. Ключ другого провайдера
    при забытом base_url — это 401 на пятой минуте вместо отказа на старте."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ключ")
    config = {"backends": {"deepseek": {"type": "openai_compat",
                                        "api_key_env": "DEEPSEEK_API_KEY"}}}

    with pytest.raises(BackendError, match="base_url"):
        build_backends(config, tmp_path)


def test_openai_compat_without_base_url_is_fine_for_openai_itself(tmp_path,
                                                                  monkeypatch):
    """Сам OpenAI живёт по умолчанию — спека описывает его именно так."""
    monkeypatch.setenv("OPENAI_API_KEY", "ключ")
    config = {"backends": {"openai": {"type": "openai_compat",
                                      "api_key_env": "OPENAI_API_KEY"}}}

    assert "openai" in build_backends(config, tmp_path)
