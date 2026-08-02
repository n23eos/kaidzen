import pytest
from pydantic import BaseModel
from kaidzen.llm import LLMClient, SchemaValidationFailure


class Out(BaseModel):
    answer: str
    score: int


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
    def __init__(self, blocks, in_tok=10, out_tok=5, searches=None):
        self.content = blocks
        self.usage = FakeUsage(in_tok, out_tok, searches)


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def make_client(responses):
    client = LLMClient(api_key="test-key")
    fake = FakeMessages(responses)
    client._client = type("FakeAnthropic", (), {"messages": fake})()
    return client, fake


def good():
    return FakeResponse([FakeBlock("tool_use", name="submit",
                                   input={"answer": "ok", "score": 7})])


def missing_field():
    return FakeResponse([FakeBlock("tool_use", name="submit",
                                   input={"answer": "ok"})])


def no_tool_call():
    return FakeResponse([FakeBlock("text", text="я просто поболтаю")])


def test_structured_returns_validated_model():
    client, _ = make_client([good()])
    out = client.structured(model="m", system="s", user="u",
                            schema=Out, temperature=0.1)
    assert out == Out(answer="ok", score=7)


def test_forces_tool_choice_without_web_search():
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out, temperature=0)
    assert fake.calls[0]["tool_choice"] == {"type": "tool", "name": "submit"}
    assert [t["name"] for t in fake.calls[0]["tools"]] == ["submit"]


def test_web_search_adds_tool_and_drops_tool_choice():
    client, fake = make_client([good()])
    client.structured(model="m", system="s", user="u", schema=Out,
                      temperature=0, web_search=True, max_searches=5)
    call = fake.calls[0]
    assert "tool_choice" not in call
    names = [t.get("name") for t in call["tools"]]
    assert "submit" in names and "web_search" in names


def test_retry_once_on_schema_error_then_success():
    client, fake = make_client([missing_field(), good()])
    out = client.structured(model="m", system="s", user="u",
                            schema=Out, temperature=0.1)
    assert out.score == 7
    assert len(fake.calls) == 2
    assert "score" in str(fake.calls[1]["messages"])


def test_retry_when_model_never_calls_submit():
    client, fake = make_client([no_tool_call(), good()])
    out = client.structured(model="m", system="s", user="u",
                            schema=Out, temperature=0)
    assert out.score == 7
    assert len(fake.calls) == 2


def test_two_schema_failures_raise():
    client, fake = make_client([missing_field(), missing_field()])
    with pytest.raises(SchemaValidationFailure):
        client.structured(model="m", system="s", user="u",
                          schema=Out, temperature=0.1)
    assert len(fake.calls) == 2


def test_usage_accumulates_across_calls():
    client, _ = make_client([good(), good()])
    client.structured(model="m", system="s", user="u", schema=Out, temperature=0)
    client.structured(model="m", system="s", user="u", schema=Out, temperature=0)
    assert client.usage.input_tokens == 20
    assert client.usage.output_tokens == 10
    assert client.usage.web_searches == 0


def test_usage_counts_web_searches():
    client, _ = make_client([FakeResponse(
        [FakeBlock("tool_use", name="submit", input={"answer": "a", "score": 1})],
        searches=3)])
    client.structured(model="m", system="s", user="u", schema=Out,
                      temperature=0, web_search=True)
    assert client.usage.web_searches == 3


def test_usage_counted_even_on_failed_attempt():
    client, _ = make_client([missing_field(), good()])
    client.structured(model="m", system="s", user="u", schema=Out, temperature=0)
    assert client.usage.input_tokens == 20  # обе попытки оплачены
