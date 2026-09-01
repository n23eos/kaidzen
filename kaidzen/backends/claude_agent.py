"""Бэкенд подписки Claude: claude_agent_sdk, без API-ключа.

Тут нет tool-use со схемой: JSON просят в системном промпте и валидируют
на выходе. Веб-поиск — встроенный инструмент WebSearch, число вызовов видно
в потоке как ToolUseBlock.
"""
from __future__ import annotations

from typing import Any, Type, TypeVar

import anyio
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError, query
from pydantic import BaseModel, ValidationError

from kaidzen.backends.base import (DEFAULT_MAX_SEARCHES, DEFAULT_MAX_TOKENS,
                                   MAX_SCHEMA_RETRIES, BackendError,
                                   LLMBackend, SchemaValidationFailure)
from kaidzen.backends.json_extract import (JsonExtractionError,
                                           extract_json_object)

T = TypeVar("T", bound=BaseModel)

WEB_SEARCH_TOOL = "WebSearch"
# Потолки, а не бюджеты: SDK тратит столько ходов, сколько нужно, и неизрасходованные
# ничего не стоят. Скупость здесь обходится дорого — на полном бенчмарке лимиты 4/12
# уронили почти треть прогонов ('Reached maximum number of turns'), а каждое падение
# выбивает идею из сравнения кандидатов. Researcher делает до трёх поисков на допущение
# при пяти допущениях за итерацию, то есть полтора десятка ходов только на запросы.
TURNS_WITHOUT_SEARCH = 12
TURNS_WITH_SEARCH = 40

JSON_INSTRUCTION = (
    "Верни СТРОГО один JSON-объект по схеме ниже. Без markdown-ограды, "
    "без пояснений до или после. Схема:\n{schema}")


class ClaudeAgentBackend(LLMBackend):
    """Прогон на подписке: платить не нужно, ограничение — лимиты Claude."""

    supports_web_search = True

    def structured(self, *, model: str, system: str, user: str,
                   schema: Type[T], effort: str,
                   web_search: bool = False,
                   max_searches: int = DEFAULT_MAX_SEARCHES,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> T:
        """Возвращает валидированный объект схемы.

        effort, max_tokens и max_searches здесь не имеют аналога: Agent SDK не
        принимает ни глубину размышления, ни потолок токенов, ни лимит поисков.
        Параметры оставлены ради единого контракта и осознанно игнорируются.
        """
        system_prompt = self._system_with_schema(system, schema)
        options = self._options(system_prompt, web_search, model)
        prompt = user
        last_error = ""
        for _ in range(MAX_SCHEMA_RETRIES + 1):
            text, searches = self._ask(prompt, options)
            try:
                parsed = schema.model_validate(extract_json_object(text))
            except (JsonExtractionError, ValidationError) as e:
                last_error = str(e)
                prompt = self._retry_prompt(user, last_error)
                continue
            self._guard_search_performed(
                web_search=web_search, searches_done=searches)
            return parsed
        raise SchemaValidationFailure(last_error)

    @staticmethod
    def _system_with_schema(system: str, schema: Type[BaseModel]) -> str:
        instruction = JSON_INSTRUCTION.format(schema=schema.model_json_schema())
        return f"{system}\n\n{instruction}"

    @staticmethod
    def _retry_prompt(user: str, error: str) -> str:
        return (f"{user}\n\nПредыдущий ответ не прошёл валидацию: {error}\n"
                f"Верни исправленный JSON-объект по схеме.")

    @staticmethod
    def _options(system_prompt: str, web_search: bool,
                 model: str) -> ClaudeAgentOptions:
        """tools ограничивает набор инструментов, allowed_tools — авторазрешение.

        Задаём оба: без tools=[] модель получила бы весь набор Claude Code.
        """
        tools = [WEB_SEARCH_TOOL] if web_search else []
        turns = TURNS_WITH_SEARCH if web_search else TURNS_WITHOUT_SEARCH
        return ClaudeAgentOptions(system_prompt=system_prompt, model=model,
                                  tools=tools, allowed_tools=list(tools),
                                  max_turns=turns)

    def _ask(self, prompt: str, options: ClaudeAgentOptions) -> tuple[str, int]:
        """Синхронный мост к асинхронному query(): текст ответа и число поисков.

        Ошибки SDK переводятся в BackendError: истёкшая сессия, отсутствующий
        `claude` в PATH и упавший процесс — это состояние окружения, а не сбой
        кода, и пользователь должен увидеть строку с подсказкой, а не трейсбек
        с внутренностями asyncio. Поймано живым прогоном на истёкшем OAuth.
        """
        try:
            return anyio.run(self._stream, prompt, options)
        except ClaudeSDKError as e:
            raise BackendError(
                f"бэкенд подписки не смог выполнить запрос: {e}. Проверьте, "
                f"что CLI claude установлен и сессия жива (claude login)") from e

    async def _stream(self, prompt: str,
                      options: ClaudeAgentOptions) -> tuple[str, int]:
        texts: list[str] = []
        searches = 0
        async for message in query(prompt=prompt, options=options):
            blocks = getattr(message, "content", None)
            if blocks is None:
                self._record_result_usage(message)
                continue
            for block in blocks:
                if getattr(block, "name", None) == WEB_SEARCH_TOOL:
                    searches += 1
                text = getattr(block, "text", None)
                if text:
                    texts.append(text)
        self.usage.web_searches += searches
        return self._pick_json_text(texts), searches

    @staticmethod
    def _pick_json_text(texts: list[str]) -> str:
        """Финальный JSON приходит последним; ранние ходы — рассуждения."""
        for text in reversed(texts):
            if "{" in text:
                return text
        return "\n".join(texts)

    def _record_usage_dict(self, usage: dict[str, Any]) -> None:
        self.usage.input_tokens += usage.get("input_tokens", 0) or 0
        self.usage.output_tokens += usage.get("output_tokens", 0) or 0

    def _record_result_usage(self, message: Any) -> None:
        """ResultMessage отличаем по отсутствию content: у него только итоги."""
        usage = getattr(message, "usage", None)
        if isinstance(usage, dict):
            self._record_usage_dict(usage)
