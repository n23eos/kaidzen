from typing import Optional

import pytest
from pydantic import BaseModel
from kaidzen.llm import (LLMClient, PauseTurnLimitExceeded, ResponseTruncated,
                         SchemaValidationFailure, MAX_PAUSE_CONTINUATIONS,
                         STREAMING_MAX_TOKENS)
from kaidzen.state import Fact

# Порог держим здесь, а не берём из kaidzen.llm: тест обязан заметить,
# если кто-то вернёт в код обрезающий бюджет вроде 4096.
MIN_SAFE_MAX_TOKENS = 16000


class Out(BaseModel):
    answer: str
    score: int


class NestedOut(BaseModel):
    """Схема с вложенной моделью и Optional — как реальные схемы ролей."""
    facts: list[Fact]
    note: Optional[str] = None


class FakeBlock:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class FakeUsage:
    def __init__(self, input_tokens, output_tokens, web_search_requests=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.server_tool_use = (
            FakeBlock("usage", web_search_requests=web_search_requests)
            if web_search_requests is not None else None)


class FakeResponse:
    def __init__(self, blocks, in_tok=10, out_tok=5, searches=None,
                 stop_reason="tool_use"):
        self.content = blocks
        self.usage = FakeUsage(in_tok, out_tok, searches)
        self.stop_reason = stop_reason


class FakeStreamManager:
    """Повторяет форму client.messages.stream(): контекст + get_final_message()."""

    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self._response


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []          # все вызовы, независимо от транспорта
        self.create_calls = []
        self.stream_calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.create_calls.append(kwargs)
        return self._responses.pop(0)

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        self.stream_calls.append(kwargs)
        return FakeStreamManager(self._responses.pop(0))


def make_client(responses):
    client = LLMClient(api_key="test-key")
    fake = FakeMessages(responses)
    client._client = type("FakeAnthropic", (), {"messages": fake})()
    return client, fake


def good(block_id="tu_ok"):
    return FakeResponse([FakeBlock("tool_use", id=block_id, name="submit",
                                   input={"answer": "ok", "score": 7})])


def missing_field(block_id="tu_bad"):
    return FakeResponse([FakeBlock("tool_use", id=block_id, name="submit",
                                   input={"answer": "ok"})])


def no_tool_call():
    return FakeResponse([FakeBlock("text", text="я просто поболтаю")],
                        stop_reason="end_turn")


def paused(text="ищу..."):
    """Ответ сервера с приостановленным ходом: submit ещё не вызван."""
    return FakeResponse([FakeBlock("text", text=text)], stop_reason="pause_turn")


def test_structured_returns_validated_model():
    client, _ = make_client([good()])
    out = client.structured(model="m", system="s", user="u",
                            schema=Out, effort="low")
    assert out == Out(answer="ok", score=7)


def test_forces_tool_choice_without_web_search():
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out, effort="low")
    assert fake.calls[0]["tool_choice"] == {"type": "tool", "name": "submit"}
    assert [t["name"] for t in fake.calls[0]["tools"]] == ["submit"]


def test_web_search_adds_tool_and_drops_tool_choice():
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out,
                      effort="low", web_search=True, max_searches=5)
    call = fake.calls[0]
    assert "tool_choice" not in call
    names = [t.get("name") for t in call["tools"]]
    assert "submit" in names and "web_search" in names


def test_retry_keeps_conversation_protocol_valid():
    """Провал валидации: сохраняем ход ассистента и отвечаем tool_result'ом."""
    failed = missing_field(block_id="tu_1")
    client, fake = make_client([failed, good()])
    out = client.structured(model="m", system="s", user="u",
                            schema=Out, effort="low")
    assert out.score == 7
    assert len(fake.calls) == 2

    messages = fake.calls[1]["messages"]
    assert len(messages) == 3
    # исходный вопрос пользователя не потерян
    assert messages[0] == {"role": "user", "content": "u"}
    # ход ассистента со всеми блоками (включая результаты веб-поиска) сохранён
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] is failed.content
    # на tool_use обязан быть ровно один tool_result с тем же id
    assert messages[2]["role"] == "user"
    result = messages[2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "tu_1"
    assert result["is_error"] is True
    assert "score" in result["content"]


def test_retry_when_model_never_calls_submit():
    """Нет tool_use — отвечать нечем, шлём обычное текстовое напоминание."""
    silent = no_tool_call()
    client, fake = make_client([silent, good()])
    out = client.structured(model="m", system="s", user="u",
                            schema=Out, effort="low")
    assert out.score == 7
    assert len(fake.calls) == 2

    messages = fake.calls[1]["messages"]
    assert messages[0] == {"role": "user", "content": "u"}
    assert messages[1] == {"role": "assistant", "content": silent.content}
    assert messages[2]["role"] == "user"
    assert isinstance(messages[2]["content"], str)
    assert "submit" in messages[2]["content"]


def test_pause_turn_continues_without_spending_schema_retry():
    """pause_turn — не ошибка схемы: продолжаем ход и сохраняем бюджет retry."""
    pause = paused()
    client, fake = make_client([pause, missing_field(), good()])
    out = client.structured(model="m", system="s", user="u", schema=Out,
                            effort="low", web_search=True)
    assert out.score == 7
    assert len(fake.calls) == 3
    # после паузы дописан ход ассистента, никакого текста про валидацию
    assert fake.calls[1]["messages"] == [{"role": "user", "content": "u"},
                                         {"role": "assistant", "content": pause.content}]


def test_pause_turn_bounded_by_max_continuations():
    responses = [paused() for _ in range(MAX_PAUSE_CONTINUATIONS + 1)]
    client, fake = make_client(responses)
    with pytest.raises(PauseTurnLimitExceeded):
        client.structured(model="m", system="s", user="u", schema=Out,
                          effort="low", web_search=True)
    assert len(fake.calls) == MAX_PAUSE_CONTINUATIONS + 1


def test_max_tokens_raises_distinct_error():
    truncated = FakeResponse([FakeBlock("text", text="дли")], stop_reason="max_tokens")
    client, fake = make_client([truncated, good()])
    with pytest.raises(ResponseTruncated):
        client.structured(model="m", system="s", user="u", schema=Out, effort="low")
    assert len(fake.calls) == 1  # обрезка — не повод повторять


def test_nested_schema_input_schema_has_defs_and_refs():
    """Реальные схемы ролей вложенные: $defs/$ref должны уходить как есть."""
    client, fake = make_client([FakeResponse([FakeBlock(
        "tool_use", id="tu_n", name="submit",
        input={"facts": [{"claim": "c", "source_url": "https://x.test"}]})])])
    out = client.structured(model="m", system="s", user="u",
                            schema=NestedOut, effort="low")
    assert out.facts[0].source_url == "https://x.test"

    sent = fake.calls[0]["tools"][0]["input_schema"]
    assert sent == NestedOut.model_json_schema()
    assert "Fact" in sent["$defs"]
    assert sent["properties"]["facts"]["items"]["$ref"] == "#/$defs/Fact"
    assert "note" in sent["properties"]


def test_two_schema_failures_raise():
    client, fake = make_client([missing_field(), missing_field()])
    with pytest.raises(SchemaValidationFailure):
        client.structured(model="m", system="s", user="u",
                          schema=Out, effort="low")
    assert len(fake.calls) == 2


def test_usage_accumulates_across_calls():
    client, _ = make_client([
        FakeResponse([FakeBlock("tool_use", id="t1", name="submit",
                                input={"answer": "ok", "score": 7})],
                     in_tok=10, out_tok=5, searches=2),
        FakeResponse([FakeBlock("tool_use", id="t2", name="submit",
                                input={"answer": "ok", "score": 7})],
                     in_tok=30, out_tok=7, searches=1)])
    client.structured(model="m", system="s", user="u", schema=Out, effort="low")
    client.structured(model="m", system="s", user="u", schema=Out, effort="low")
    assert client.usage.input_tokens == 40
    assert client.usage.output_tokens == 12
    assert client.usage.web_searches == 3


def test_usage_counts_web_searches():
    client, _ = make_client([FakeResponse(
        [FakeBlock("tool_use", name="submit", input={"answer": "a", "score": 1})],
        searches=3)])
    client.structured(model="m", system="s", user="u", schema=Out,
                      effort="low", web_search=True)
    assert client.usage.web_searches == 3


def test_usage_counted_even_on_failed_attempt():
    client, _ = make_client([missing_field(), good()])
    client.structured(model="m", system="s", user="u", schema=Out, effort="low")
    assert client.usage.input_tokens == 20  # обе попытки оплачены


# --- контракт запроса ------------------------------------------------------
# Фейки принимают **kwargs и молча съедают любой лишний параметр, поэтому
# форму запроса проверяем явно: иначе зелёные тесты не заметят 400 от API.

@pytest.mark.parametrize("forbidden", ["temperature", "top_p", "top_k"])
def test_request_carries_no_sampling_params(forbidden):
    """На claude-sonnet-5 любой из этих параметров даёт 400 invalid_request."""
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out, effort="low")
    assert forbidden not in fake.calls[0]


def test_effort_goes_inside_output_config():
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out, effort="high")
    assert fake.calls[0]["output_config"] == {"effort": "high"}
    assert "effort" not in fake.calls[0]  # effort — не параметр верхнего уровня


def test_default_max_tokens_leaves_room_for_thinking():
    """Adaptive thinking включено по умолчанию и ест тот же бюджет, что и текст."""
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out, effort="low")
    assert fake.calls[0]["max_tokens"] >= MIN_SAFE_MAX_TOKENS


def test_web_search_tool_type_is_current():
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out,
                      effort="low", web_search=True)
    web = [t for t in fake.calls[0]["tools"] if t.get("name") == "web_search"][0]
    assert web["type"] == "web_search_20260209"


def test_no_code_execution_tool_alongside_web_search():
    """Динамическая фильтрация сама запускает код: второе окружение мешает."""
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out,
                      effort="low", web_search=True)
    types = [t.get("type", "") for t in fake.calls[0]["tools"]]
    assert not any(t.startswith("code_execution") for t in types)


def test_small_max_tokens_uses_create():
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out, effort="low",
                      max_tokens=STREAMING_MAX_TOKENS - 1)
    assert len(fake.create_calls) == 1
    assert fake.stream_calls == []


def test_large_max_tokens_uses_stream():
    """Большой max_tokens без стрима рвёт HTTP-соединение по таймауту SDK."""
    client, fake = make_client([good()])
    out = client.structured(model="m", system="s", user="u", schema=Out,
                            effort="low", max_tokens=STREAMING_MAX_TOKENS)
    assert out.score == 7
    assert len(fake.stream_calls) == 1
    assert fake.create_calls == []
