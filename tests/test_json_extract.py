"""Извлечение JSON-объекта из свободного текста модели."""
import pytest

from kaidzen.backends.json_extract import (JsonExtractionError,
                                           extract_json_object)


def test_plain_object():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_fence_with_language_tag():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_fence_without_language_tag():
    assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_prose_around_object():
    assert extract_json_object('Вот: {"a": 1}. Готово.') == {"a": 1}


def test_text_without_object_raises():
    with pytest.raises(JsonExtractionError):
        extract_json_object("никакого JSON тут нет")


def test_json_array_is_not_an_object():
    with pytest.raises(JsonExtractionError):
        extract_json_object("[1, 2, 3]")
