"""Обёртка Anthropic SDK: structured output через tool-use, retry, учёт usage."""
from __future__ import annotations

from typing import Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from kaidzen.state import ApiUsage

T = TypeVar("T", bound=BaseModel)

SUBMIT_TOOL = "submit"
MAX_SCHEMA_RETRIES = 1      # один повтор с текстом ошибки валидации
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_SEARCHES = 8


class SchemaValidationFailure(Exception):
    """Модель дважды не смогла вернуть ответ по схеме."""


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self._client = (anthropic.Anthropic(api_key=api_key) if api_key
                        else anthropic.Anthropic())
        self.usage = ApiUsage()

    def structured(self, *, model: str, system: str, user: str,
                   schema: Type[T], temperature: float,
                   web_search: bool = False,
                   max_searches: int = DEFAULT_MAX_SEARCHES,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> T:
        """Вызывает модель и возвращает валидированный объект схемы."""
        tools: list[dict] = [self._submit_tool(schema)]
        extra: dict = {}
        if web_search:
            # с web_search нельзя форсировать tool_choice: модель должна
            # сначала свободно поискать, и только потом вызвать submit
            tools.append({"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search",
                          "max_uses": max_searches})
        else:
            extra["tool_choice"] = {"type": "tool", "name": SUBMIT_TOOL}

        messages = [{"role": "user", "content": user}]
        last_error = ""
        for _ in range(MAX_SCHEMA_RETRIES + 1):
            if last_error:
                messages = messages + [{
                    "role": "user",
                    "content": (f"Твой прошлый ответ не прошёл валидацию: {last_error}. "
                                f"Вызови tool '{SUBMIT_TOOL}' ещё раз с исправленными полями.")}]
            response = self._client.messages.create(
                model=model, system=system, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                tools=tools, **extra)
            self._record_usage(response)
            payload = self._extract_submit(response)
            if payload is None:
                last_error = f"ответ не содержит вызова tool '{SUBMIT_TOOL}'"
                continue
            try:
                return schema.model_validate(payload)
            except ValidationError as e:
                last_error = str(e)
        raise SchemaValidationFailure(last_error)

    def _submit_tool(self, schema: Type[BaseModel]) -> dict:
        return {"name": SUBMIT_TOOL,
                "description": "Отправить финальный структурированный ответ.",
                "input_schema": schema.model_json_schema()}

    def _extract_submit(self, response) -> dict | None:
        for block in response.content:
            if block.type == "tool_use" and block.name == SUBMIT_TOOL:
                return block.input
        return None

    def _record_usage(self, response) -> None:
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens
        server_tools = getattr(response.usage, "server_tool_use", None)
        if server_tools is not None:
            self.usage.web_searches += getattr(server_tools, "web_search_requests", 0) or 0
