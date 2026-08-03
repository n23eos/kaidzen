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


def test_raw_newline_inside_string_is_tolerated():
    """Модель ставит настоящий перевод строки внутрь значения — поймано вживую:
    'Invalid control character at: line 1 column 2703'."""
    text = '{"claim":"первая строка\nвторая строка","n":1}'
    assert extract_json_object(text)["n"] == 1


def test_raw_tab_inside_string_is_tolerated():
    assert extract_json_object('{"a":"до\tпосле"}')["a"] == "до\tпосле"
