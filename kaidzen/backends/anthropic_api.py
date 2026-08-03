"""Бэкенд Anthropic API: structured output через tool-use, retry, учёт usage."""
from __future__ import annotations

from typing import Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from kaidzen.backends.base import (DEFAULT_MAX_SEARCHES, DEFAULT_MAX_TOKENS,
                                   MAX_SCHEMA_RETRIES, LLMBackend,
                                   PauseTurnLimitExceeded, ResponseTruncated,
                                   SchemaValidationFailure)

T = TypeVar("T", bound=BaseModel)

SUBMIT_TOOL = "submit"
# pause_turn — это не ошибка: сервер приостановил долгий ход (веб-поиск).
# Продолжаем ход, но ограничиваем число продолжений, чтобы не крутиться вечно.
MAX_PAUSE_CONTINUATIONS = 4
# версия с динамической фильтрацией: результаты поиска отсеиваются ДО того,
# как попадут в контекст — критично для Researcher с восемью поисками за вызов
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"
# выше этого потолка SDK требует стриминга, иначе рвёт HTTP по таймауту
STREAMING_MAX_TOKENS = 16000


class AnthropicApiBackend(LLMBackend):
    """Платный транспорт: ключ ANTHROPIC_API_KEY, серверный веб-поиск."""

    supports_web_search = True

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self._client = (anthropic.Anthropic(api_key=api_key) if api_key
                        else anthropic.Anthropic())

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
        tools, extra = self._build_tools(schema, web_search, max_searches)
        messages: list[dict] = [{"role": "user", "content": user}]
        searches_done = 0
        schema_failures = 0
        pause_continuations = 0
        while True:
            response = self._send(
                model=model, system=system, messages=messages,
                output_config={"effort": effort}, max_tokens=max_tokens,
                tools=tools, **extra)
            searches_done += self._record_usage(response)
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
            if block is not None:
                try:
                    parsed = schema.model_validate(block.input)
                except ValidationError as e:
                    last_error, feedback = str(e), self._schema_feedback(block, e)
                else:
                    self._guard_search_performed(
                        web_search=web_search, searches_done=searches_done)
                    return parsed
            else:
                last_error, feedback = self._missing_submit_feedback()

            schema_failures += 1
            if schema_failures > MAX_SCHEMA_RETRIES:
                raise SchemaValidationFailure(last_error)
            # сохраняем ход ассистента целиком: там же лежат результаты
            # веб-поиска, которые иначе пришлось бы искать заново
            messages = messages + [self._assistant_turn(response), feedback]

    def _build_tools(self, schema: Type[BaseModel], web_search: bool,
                     max_searches: int) -> tuple[list[dict], dict]:
        tools: list[dict] = [self._submit_tool(schema)]
        if web_search:
            # с web_search нельзя форсировать tool_choice: модель должна
            # сначала свободно поискать, и только потом вызвать submit
            tools.append({"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search",
                          "max_uses": max_searches})
            return tools, {}
        return tools, {"tool_choice": {"type": "tool", "name": SUBMIT_TOOL}}

    @staticmethod
    def _schema_feedback(block, error: ValidationError) -> dict:
        """На каждый tool_use обязан быть tool_result с тем же id."""
        return {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": block.id,
            "is_error": True,
            "content": (f"Ответ не прошёл валидацию: {error}. "
                        f"Вызови '{SUBMIT_TOOL}' ещё раз с исправленными полями.")}]}

    @staticmethod
    def _missing_submit_feedback() -> tuple[str, dict]:
        """tool_use не было — отвечать нечем, шлём обычное напоминание."""
        return (f"ответ не содержит вызова tool '{SUBMIT_TOOL}'",
                {"role": "user",
                 "content": (f"Ты не вызвал tool '{SUBMIT_TOOL}'. "
                             f"Вызови его и передай ответ по схеме.")})

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

    @staticmethod
    def _submit_tool(schema: Type[BaseModel]) -> dict:
        return {"name": SUBMIT_TOOL,
                "description": "Отправить финальный структурированный ответ.",
                "input_schema": schema.model_json_schema()}

    @staticmethod
    def _extract_submit(response):
        """Возвращает сам блок tool_use: нужен и payload, и его id для tool_result."""
        for block in response.content:
            if block.type == "tool_use" and block.name == SUBMIT_TOOL:
                return block
        return None

    def _record_usage(self, response) -> int:
        """Копит usage и возвращает число поисков в этом ответе."""
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens
        server_tools = getattr(response.usage, "server_tool_use", None)
        if server_tools is None:
            return 0
        searches = getattr(server_tools, "web_search_requests", 0) or 0
        self.usage.web_searches += searches
        return searches
