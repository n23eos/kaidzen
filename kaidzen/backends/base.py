"""Общий контракт бэкендов моделей: исключения, способности, учёт usage."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

from kaidzen.state import ApiUsage

T = TypeVar("T", bound=BaseModel)

# adaptive thinking на claude-sonnet-5 включено по умолчанию, а max_tokens —
# общий потолок на размышление И на текст ответа, поэтому нужен запас
DEFAULT_MAX_TOKENS = 16000
DEFAULT_MAX_SEARCHES = 8
MAX_SCHEMA_RETRIES = 1      # один повтор с текстом ошибки валидации


class SchemaValidationFailure(Exception):
    """Модель дважды не смогла вернуть ответ по схеме."""


class ResponseTruncated(Exception):
    """Ответ обрезан по max_tokens — это не проблема схемы, повтор не поможет."""


class PauseTurnLimitExceeded(Exception):
    """Слишком много продолжений приостановленного хода."""


class SearchNotPerformed(ValueError):
    """Поиск был запрошен, но бэкенд не выполнил ни одного запроса.

    Наследуется от ValueError намеренно: ValueError уже входит в
    RETRYABLE_ERRORS оркестратора, поэтому шаг перезапросится сам, без
    правки списка retry.

    Зачем вообще падать: без поиска модель охотно возвращает правдоподобный,
    но выдуманный URL из памяти. Такой источник отравляет всю ось
    обоснованности — лучше повторить шаг, чем записать выдумку в отчёт.
    """


class BackendError(Exception):
    """Ошибка конфигурации или инициализации бэкенда (ключ, тип, base_url)."""


class LLMBackend(ABC):
    """Транспорт до модели: один вызов — один валидированный объект схемы.

    Способности объявляются классом (или экземпляром) и читаются валидатором
    конфига ДО первого платного вызова: например, роль researcher нельзя
    поставить на бэкенд без веб-поиска.
    """

    # объявляем честно: True только если поиск реально реализован в вызове
    supports_web_search: bool = False

    def __init__(self) -> None:
        self.usage = ApiUsage()

    @abstractmethod
    def structured(self, *, model: str, system: str, user: str,
                   schema: Type[T], effort: str,
                   web_search: bool = False,
                   max_searches: int = DEFAULT_MAX_SEARCHES,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> T:
        """Вызывает модель и возвращает валидированный объект схемы."""

    def _guard_search_performed(self, *, web_search: bool,
                                searches_done: int) -> None:
        """Гарантия §4 ТЗ: запрошенный поиск обязан был реально произойти."""
        if web_search and searches_done <= 0:
            raise SearchNotPerformed(
                "поиск был запрошен, но не выполнен ни разу: "
                "ссылки из памяти модели непроверяемы")

    def __repr__(self) -> str:
        """Без единого поля с секретами: ключи не должны утекать в логи."""
        return f"<{type(self).__name__} web_search={self.supports_web_search}>"
