"""Бэкенд openai_compat: два режима structured output, повтор, usage."""
import pytest
from pydantic import BaseModel

from kaidzen.backends.base import BackendError, SchemaValidationFailure
from kaidzen.backends.openai_compat import (MODE_JSON_OBJECT, MODE_JSON_SCHEMA,
                                            OpenAICompatBackend, strict_schema)


class Out(BaseModel):
    answer: str
    score: int


GOOD_JSON = '{"answer":"ok","score":7}'
BAD_JSON = '{"answer":"ok"}'


class FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeChoice:
    def __init__(self, content):
        self.message = type("Msg", (), {"content": content})()


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


class FakeCompletions:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self._contents.pop(0))


class FakeClient:
    def __init__(self, contents):
        self.completions = FakeCompletions(contents)
        self.chat = self

    @property
    def calls(self):
        return self.completions.calls


def make_backend(contents, mode=MODE_JSON_SCHEMA):
    client = FakeClient(contents)
    backend = OpenAICompatBackend(base_url="https://example.test",
                                  api_key="sk-secret-value",
                                  structured_mode=mode, client=client)
    return backend, client


def call(backend, **kw):
    return backend.structured(model="m", system="s", user="u", schema=Out,
                              effort="high", **kw)


def test_happy_path_returns_validated_model():
    backend, _ = make_backend([GOOD_JSON])
    assert call(backend) == Out(answer="ok", score=7)


def test_openai_mode_sends_strict_json_schema():
    backend, client = make_backend([GOOD_JSON])
    call(backend)
    fmt = client.calls[0]["response_format"]
    assert fmt["type"] == MODE_JSON_SCHEMA
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    # схему не дублируем в промпт: её знает сам API
    assert client.calls[0]["messages"][0]["content"] == "s"


def test_deepseek_mode_sends_json_object_and_schema_in_prompt():
    backend, client = make_backend([GOOD_JSON], mode=MODE_JSON_OBJECT)
    call(backend)
    assert client.calls[0]["response_format"] == {"type": MODE_JSON_OBJECT}
    assert "score" in client.calls[0]["messages"][0]["content"]


def test_retry_after_validation_failure_then_success():
    backend, client = make_backend([BAD_JSON, GOOD_JSON])
    assert call(backend).score == 7
    assert len(client.calls) == 2
    messages = client.calls[1]["messages"]
    assert messages[-2] == {"role": "assistant", "content": BAD_JSON}
    assert "валидацию" in messages[-1]["content"]


def test_two_failures_raise():
    backend, client = make_backend([BAD_JSON, BAD_JSON])
    with pytest.raises(SchemaValidationFailure):
        call(backend)
    assert len(client.calls) == 2


def test_fenced_response_is_parsed():
    backend, _ = make_backend([f"```json\n{GOOD_JSON}\n```"], MODE_JSON_OBJECT)
    assert call(backend).score == 7


def test_usage_accumulates():
    backend, _ = make_backend([GOOD_JSON, GOOD_JSON])
    call(backend)
    call(backend)
    assert backend.usage.input_tokens == 20
    assert backend.usage.output_tokens == 10


def test_web_search_is_rejected_because_not_implemented():
    backend, _ = make_backend([GOOD_JSON])
    with pytest.raises(BackendError):
        call(backend, web_search=True)


def test_declares_no_web_search_capability():
    """Флагу верит валидатор конфига: поиск здесь не реализован."""
    assert OpenAICompatBackend.supports_web_search is False


def test_unknown_mode_rejected():
    with pytest.raises(BackendError):
        OpenAICompatBackend(structured_mode="magic", client=FakeClient([]))


def test_api_key_never_appears_in_repr():
    backend, _ = make_backend([GOOD_JSON])
    assert "sk-secret-value" not in repr(backend)


def test_strict_schema_marks_nested_objects():
    """Strict-режим требует запрета лишних полей на каждом уровне."""
    source = {"type": "object",
              "properties": {"inner": {"type": "object",
                                       "properties": {"a": {"type": "string"}}}}}
    result = strict_schema(source)
    assert result["additionalProperties"] is False
    assert result["properties"]["inner"]["additionalProperties"] is False
    assert result["properties"]["inner"]["required"] == ["a"]
    # исходная схема не изменена
    assert "additionalProperties" not in source


def test_response_without_usage_is_tolerated():
    backend, client = make_backend([GOOD_JSON])
    monkey = client.completions.create

    def create_without_usage(**kwargs):
        response = monkey(**kwargs)
        response.usage = None
        return response

    client.completions.create = create_without_usage
    assert call(backend).score == 7
    assert backend.usage.input_tokens == 0
