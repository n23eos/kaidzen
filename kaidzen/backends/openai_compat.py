"""Бэкенд OpenAI-совместимого API: сам OpenAI и DeepSeek.

Одна реализация, две конфигурации. Различие — не в URL, а в объявленном
режиме structured output: OpenAI умеет strict json_schema, DeepSeek — только
json_object, поэтому ему схему кладут в промпт.
"""
from __future__ import annotations

from typing import Any, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from kaidzen.backends.base import (DEFAULT_MAX_SEARCHES, DEFAULT_MAX_TOKENS,
                                   MAX_SCHEMA_RETRIES, BackendError,
                                   LLMBackend, SchemaValidationFailure)
from kaidzen.backends.json_extract import (JsonExtractionError,
                                           extract_json_object)

T = TypeVar("T", bound=BaseModel)

MODE_JSON_SCHEMA = "json_schema"     # OpenAI: строгая схема на стороне API
MODE_JSON_OBJECT = "json_object"     # DeepSeek: только «верни любой JSON»
STRUCTURED_MODES = (MODE_JSON_SCHEMA, MODE_JSON_OBJECT)

SCHEMA_IN_PROMPT = (
    "Верни СТРОГО один JSON-объект по схеме ниже, без пояснений.\nСхема:\n{schema}")


class OpenAICompatBackend(LLMBackend):
    """Транспорт по протоколу chat.completions.

    Веб-поиск здесь НЕ реализован ни для одного провайдера, поэтому
    supports_web_search=False: валидатор конфига верит этому флагу и не
    поставит на такой бэкенд роль researcher.
    """

    supports_web_search = False

    def __init__(self, *, base_url: str | None = None, api_key: str = "",
                 structured_mode: str = MODE_JSON_SCHEMA,
                 client: Any = None):
        super().__init__()
        if structured_mode not in STRUCTURED_MODES:
            raise BackendError(
                f"неизвестный structured_mode: {structured_mode}")
        self._mode = structured_mode
        # client внедряется в тестах; ключ живёт только внутри клиента
        self._client = client or OpenAI(api_key=api_key, base_url=base_url)

    def structured(self, *, model: str, system: str, user: str,
                   schema: Type[T], effort: str,
                   web_search: bool = False,
                   max_searches: int = DEFAULT_MAX_SEARCHES,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> T:
        """Возвращает валидированный объект схемы.

        effort игнорируется: у chat.completions нет аналога output_config.
        """
        if web_search:
            raise BackendError(
                "веб-поиск не реализован для openai_compat-бэкендов")
        messages = [{"role": "system", "content": self._system(system, schema)},
                    {"role": "user", "content": user}]
        last_error = ""
        for _ in range(MAX_SCHEMA_RETRIES + 1):
            text = self._send(model, messages, schema, max_tokens)
            try:
                return schema.model_validate(extract_json_object(text))
            except (JsonExtractionError, ValidationError) as e:
                last_error = str(e)
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user",
                                        "content": self._retry_note(last_error)}]
        raise SchemaValidationFailure(last_error)

    def _system(self, system: str, schema: Type[BaseModel]) -> str:
        """В json_object-режиме схему знает только промпт."""
        if self._mode == MODE_JSON_SCHEMA:
            return system
        return f"{system}\n\n{SCHEMA_IN_PROMPT.format(schema=schema.model_json_schema())}"

    @staticmethod
    def _retry_note(error: str) -> str:
        return (f"Ответ не прошёл валидацию: {error}. "
                f"Верни исправленный JSON-объект по схеме.")

    def _send(self, model: str, messages: list[dict],
              schema: Type[BaseModel], max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            response_format=self._response_format(schema))
        self._record_usage(response)
        return response.choices[0].message.content or ""

    def _response_format(self, schema: Type[BaseModel]) -> dict:
        if self._mode == MODE_JSON_OBJECT:
            return {"type": MODE_JSON_OBJECT}
        return {"type": MODE_JSON_SCHEMA,
                "json_schema": {"name": schema.__name__,
                                "schema": strict_schema(schema.model_json_schema()),
                                "strict": True}}

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.usage.output_tokens += getattr(usage, "completion_tokens", 0) or 0


def strict_schema(schema: dict) -> dict:
    """Приводит JSON Schema от pydantic к требованиям strict-режима OpenAI.

    Strict требует, чтобы у каждого объекта были запрещены лишние поля и
    перечислены ВСЕ свойства в required. Возвращает новый словарь: исходную
    схему не трогаем.
    """
    if not isinstance(schema, dict):
        return schema
    result = {key: _strict_value(key, value) for key, value in schema.items()}
    if result.get("type") == "object" and "properties" in result:
        result["additionalProperties"] = False
        result["required"] = list(result["properties"].keys())
    return result


def _strict_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return ({name: strict_schema(sub) for name, sub in value.items()}
                if key in ("properties", "$defs") else strict_schema(value))
    if isinstance(value, list):
        return [strict_schema(item) for item in value]
    return value
