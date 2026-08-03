"""Достаём один JSON-объект из свободного текста модели.

Нужно бэкендам без tool-use со схемой (подписка, DeepSeek): там модель
просят вернуть JSON, но она регулярно оборачивает его в markdown-ограду
или предваряет вежливой фразой.
"""
from __future__ import annotations

import json
from typing import Any

FENCE = "```"


class JsonExtractionError(ValueError):
    """В тексте ответа не нашлось разбираемого JSON-объекта."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Возвращает первый JSON-объект верхнего уровня из текста модели."""
    candidate = _strip_fence(text.strip())
    try:
        return _load_object(candidate)
    except JsonExtractionError:
        # текст с прозой до/после: вырезаем по внешним фигурным скобкам
        return _load_object(_slice_outer_braces(candidate))


def _strip_fence(text: str) -> str:
    """Снимает markdown-ограду ```json ... ``` если она есть."""
    if not text.startswith(FENCE):
        return text
    body = text[len(FENCE):]
    newline = body.find("\n")
    if newline != -1 and not body[:newline].strip().startswith("{"):
        body = body[newline + 1:]      # отбрасываем языковой тег (```json)
    end = body.rfind(FENCE)
    return (body[:end] if end != -1 else body).strip()


def _slice_outer_braces(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise JsonExtractionError("в ответе нет JSON-объекта")
    return text[start:end + 1]


def _load_object(text: str) -> dict[str, Any]:
    try:
        # strict=False пропускает настоящие переводы строк и табы внутри
        # строковых значений. Модель их ставит регулярно — на живом прогоне
        # так потерялась целая идея бенчмарка. Смысла в отказе нет: экранирование
        # переноса нас не интересует, интересует содержимое.
        value = json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        raise JsonExtractionError(f"ответ не разбирается как JSON: {e}") from e
    if not isinstance(value, dict):
        raise JsonExtractionError("ожидался JSON-объект, пришёл другой тип")
    return value
