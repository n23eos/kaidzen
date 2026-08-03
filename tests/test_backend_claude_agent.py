"""Бэкенд подписки: разбор JSON, повтор по схеме, гарантия поиска."""
import pytest
from pydantic import BaseModel

from kaidzen.backends import claude_agent
from kaidzen.backends.base import SchemaValidationFailure, SearchNotPerformed
from kaidzen.backends.claude_agent import ClaudeAgentBackend


class Out(BaseModel):
    answer: str
    score: int


GOOD_JSON = '{"answer":"ok","score":7}'
BAD_JSON = '{"answer":"ok"}'


class FakeText:
    def __init__(self, text):
        self.text = text


class FakeToolUse:
    def __init__(self, name):
        self.name = name
        self.id = "tu_1"
        self.input = {}


class FakeAssistantMessage:
    def __init__(self, blocks):
        self.content = blocks


class FakeResultMessage:
    """У ResultMessage нет content — только итоговый usage."""

    def __init__(self, input_tokens=10, output_tokens=5):
        self.usage = {"input_tokens": input_tokens,
                      "output_tokens": output_tokens}


def make_backend(monkeypatch, turns):
    """turns — список списков сообщений: по одному списку на вызов query()."""
    calls = []
    pending = list(turns)

    async def fake_query(*, prompt, options):
        calls.append({"prompt": prompt, "options": options})
        for message in pending.pop(0):
            yield message

    monkeypatch.setattr(claude_agent, "query", fake_query)
    return ClaudeAgentBackend(), calls


def answer(text, searches=0):
    """Один ход: поисковые tool_use-блоки плюс финальный текст."""
    blocks = [FakeToolUse("WebSearch") for _ in range(searches)]
    return [FakeAssistantMessage(blocks + [FakeText(text)]),
            FakeResultMessage()]


def call(backend, **kw):
    return backend.structured(model="claude-sonnet-5", system="s", user="u",
                              schema=Out, effort="high", **kw)


def test_happy_path_returns_validated_model(monkeypatch):
    backend, calls = make_backend(monkeypatch, [answer(GOOD_JSON)])
    assert call(backend) == Out(answer="ok", score=7)
    assert len(calls) == 1


def test_schema_is_passed_in_system_prompt(monkeypatch):
    backend, calls = make_backend(monkeypatch, [answer(GOOD_JSON)])
    call(backend)
    system_prompt = calls[0]["options"].system_prompt
    assert "score" in system_prompt and "JSON" in system_prompt


def test_retry_after_validation_failure_then_success(monkeypatch):
    backend, calls = make_backend(
        monkeypatch, [answer(BAD_JSON), answer(GOOD_JSON)])
    assert call(backend).score == 7
    assert len(calls) == 2
    # во второй промпт добавлен текст ошибки валидации
    assert "валидацию" in calls[1]["prompt"]
    assert calls[1]["prompt"].startswith("u")


def test_two_failures_raise(monkeypatch):
    backend, calls = make_backend(
        monkeypatch, [answer(BAD_JSON), answer(BAD_JSON)])
    with pytest.raises(SchemaValidationFailure):
        call(backend)
    assert len(calls) == 2


def test_markdown_fence_and_prose_are_stripped(monkeypatch):
    fenced = f'Вот ответ:\n```json\n{GOOD_JSON}\n```\nГотово.'
    backend, _ = make_backend(monkeypatch, [answer(fenced)])
    assert call(backend).score == 7


def test_leading_prose_without_fence_is_stripped(monkeypatch):
    backend, _ = make_backend(
        monkeypatch, [answer(f'Конечно! {GOOD_JSON}')])
    assert call(backend).score == 7


def test_usage_accumulates_across_calls(monkeypatch):
    backend, _ = make_backend(
        monkeypatch, [answer(GOOD_JSON), answer(GOOD_JSON)])
    call(backend)
    call(backend)
    assert backend.usage.input_tokens == 20
    assert backend.usage.output_tokens == 10


def test_web_search_without_searches_raises(monkeypatch):
    backend, _ = make_backend(monkeypatch, [answer(GOOD_JSON, searches=0)])
    with pytest.raises(SearchNotPerformed):
        call(backend, web_search=True)


def test_search_not_performed_is_retryable_by_orchestrator():
    """Оркестратор повторяет ValueError — специально наследуемся от него."""
    from kaidzen.orchestrator import RETRYABLE_ERRORS
    assert issubclass(SearchNotPerformed, RETRYABLE_ERRORS)


def test_web_search_with_searches_succeeds_and_counts(monkeypatch):
    backend, calls = make_backend(monkeypatch, [answer(GOOD_JSON, searches=3)])
    assert call(backend, web_search=True).score == 7
    assert backend.usage.web_searches == 3
    assert calls[0]["options"].allowed_tools == ["WebSearch"]
    assert calls[0]["options"].max_turns > 1


def test_no_tools_and_single_turn_without_search(monkeypatch):
    backend, calls = make_backend(monkeypatch, [answer(GOOD_JSON)])
    call(backend)
    assert calls[0]["options"].allowed_tools == []
    assert calls[0]["options"].tools == []
    assert calls[0]["options"].max_turns == 1


def test_declares_web_search_capability():
    assert ClaudeAgentBackend.supports_web_search is True


def test_prose_only_answer_fails_as_schema_failure(monkeypatch):
    """Ни одного JSON за две попытки — это провал схемы, а не молчание."""
    backend, calls = make_backend(
        monkeypatch, [answer("извини, не могу"), answer("всё ещё не могу")])
    with pytest.raises(SchemaValidationFailure):
        call(backend)
    assert len(calls) == 2
