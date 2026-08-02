"""Обёртка Anthropic SDK: structured output через tool-use, retry, учёт usage."""
from __future__ import annotations

from typing import Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from kaidzen.state import ApiUsage

T = TypeVar("T", bound=BaseModel)

SUBMIT_TOOL = "submit"
MAX_SCHEMA_RETRIES = 1      # один повтор с текстом ошибки валидации
# pause_turn — это не ошибка: сервер приостановил долгий ход (веб-поиск).
# Продолжаем ход, но ограничиваем число продолжений, чтобы не крутиться вечно.
MAX_PAUSE_CONTINUATIONS = 4
# версия с динамической фильтрацией: результаты поиска отсеиваются ДО того,
# как попадут в контекст — критично для Researcher с восемью поисками за вызов
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"
# adaptive thinking на claude-sonnet-5 включено по умолчанию, а max_tokens —
# общий потолок на размышление И на текст ответа, поэтому нужен запас
DEFAULT_MAX_TOKENS = 16000
DEFAULT_MAX_SEARCHES = 8
# выше этого потолка SDK требует стриминга, иначе рвёт HTTP по таймауту
STREAMING_MAX_TOKENS = 16000


class SchemaValidationFailure(Exception):
    """Модель дважды не смогла вернуть ответ по схеме."""


class ResponseTruncated(Exception):
    """Ответ обрезан по max_tokens — это не проблема схемы, повтор не поможет."""


class PauseTurnLimitExceeded(Exception):
    """Слишком много продолжений приостановленного хода."""


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self._client = (anthropic.Anthropic(api_key=api_key) if api_key
                        else anthropic.Anthropic())
        self.usage = ApiUsage()

    def structured(self, *, model: str, system: str, user: str,
                   schema: Type[T], effort: str,
                   web_search: bool = False,
                   max_searches: int = DEFAULT_MAX_SEARCHES,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> T:
        """Вызывает модель и возвращает валидированный объект схемы.

        effort — глубина работы модели ("low"|"medium"|"high"|"xhigh"|"max").
        На claude-sonnet-5 temperature/top_p/top_k убраны: любое их значение
        возвращает 400, глубина задаётся только через output_config.
        """
        tools: list[dict] = [self._submit_tool(schema)]
        extra: dict = {}
        if web_search:
            # с web_search нельзя форсировать tool_choice: модель должна
            # сначала свободно поискать, и только потом вызвать submit
            tools.append({"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search",
                          "max_uses": max_searches})
        else:
            extra["tool_choice"] = {"type": "tool", "name": SUBMIT_TOOL}

        messages: list[dict] = [{"role": "user", "content": user}]
        schema_failures = 0
        pause_continuations = 0
        while True:
            response = self._send(
                model=model, system=system, messages=messages,
                output_config={"effort": effort}, max_tokens=max_tokens,
                tools=tools, **extra)
            self._record_usage(response)
            stop_reason = getattr(response, "stop_reason", None)

            if stop_reason == "max_tokens":
                raise ResponseTruncated(
                    f"ответ обрезан по max_tokens={max_tokens}")

            if stop_reason == "pause_turn":
                # долгий серверный tool-use: продолжаем тот же ход,
                # бюджет retry по схеме при этом не тратится
                pause_continuations += 1
                if pause_continuations > MAX_PAUSE_CONTINUATIONS:
                    raise PauseTurnLimitExceeded(
                        f"превышен лимит продолжений хода: {MAX_PAUSE_CONTINUATIONS}")
                messages = messages + [self._assistant_turn(response)]
                continue

            block = self._extract_submit(response)
            if block is None:
                last_error = f"ответ не содержит вызова tool '{SUBMIT_TOOL}'"
                # отвечать нечем: tool_use не было, шлём обычное напоминание
                feedback = {
                    "role": "user",
                    "content": (f"Ты не вызвал tool '{SUBMIT_TOOL}'. "
                                f"Вызови его и передай ответ по схеме.")}
            else:
                try:
                    return schema.model_validate(block.input)
                except ValidationError as e:
                    last_error = str(e)
                    # на каждый tool_use обязан быть tool_result с тем же id,
                    # иначе диалог невалиден для Messages API
                    feedback = {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "is_error": True,
                        "content": (f"Ответ не прошёл валидацию: {last_error}. "
                                    f"Вызови '{SUBMIT_TOOL}' ещё раз с исправленными полями.")}]}

            schema_failures += 1
            if schema_failures > MAX_SCHEMA_RETRIES:
                raise SchemaValidationFailure(last_error)
            # сохраняем ход ассистента целиком: там же лежат результаты
            # веб-поиска, которые иначе пришлось бы искать заново
            messages = messages + [self._assistant_turn(response), feedback]

    def _send(self, **kwargs):
        """Один запрос к API: большой бюджет токенов требует стриминга.

        Без стрима долгий ответ упирается в HTTP-таймаут SDK и падает,
        поэтому от порога STREAMING_MAX_TOKENS переключаемся на stream().
        """
        if kwargs["max_tokens"] < STREAMING_MAX_TOKENS:
            return self._client.messages.create(**kwargs)
        with self._client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    @staticmethod
    def _assistant_turn(response) -> dict:
        return {"role": "assistant", "content": response.content}

    def _submit_tool(self, schema: Type[BaseModel]) -> dict:
        return {"name": SUBMIT_TOOL,
                "description": "Отправить финальный структурированный ответ.",
                "input_schema": schema.model_json_schema()}

    def _extract_submit(self, response):
        """Возвращает сам блок tool_use: нужен и payload, и его id для tool_result."""
        for block in response.content:
            if block.type == "tool_use" and block.name == SUBMIT_TOOL:
                return block
        return None

    def _record_usage(self, response) -> None:
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens
        server_tools = getattr(response.usage, "server_tool_use", None)
        if server_tools is not None:
            self.usage.web_searches += getattr(server_tools, "web_search_requests", 0) or 0
