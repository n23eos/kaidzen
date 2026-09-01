"""Порождение кандидата-потомка.

Кандидаты иммутабельны: правка — это новая папка со ссылкой на родителя.
Откат тривиален, потому что прежний чемпион остался на диске нетронутым.

Запись атомарна по смыслу: невалидный потомок не остаётся на диске, иначе
следующий прогон подхватит мусор, сгенерированный мутатором.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from kaidzen.candidate import ROLES, load_candidate

META_FILE = "meta.json"
CONFIG_FILE = "config.yaml"
PROMPTS_DIR = "prompts"
JSON_INDENT = 2
INITIAL_GENERATION = 0
PENDING_STATUS = "pending"


class CandidatePatch(BaseModel):
    """Что мутатор меняет у родителя. Пустой патч = точная копия."""

    config: dict[str, Any] = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""


def _deep_merge(base: dict, patch: dict) -> dict:
    """Слияние вложенных словарей: соседние ключи секции не теряются."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_meta(candidate_dir: Path) -> dict:
    """meta.json кандидата. Битый файл — ошибка с именем файла, а не трейсбек.

    Начать с пустого словаря нельзя: тогда потомок молча стал бы корнем линии,
    а set_status затёр бы остаток файла. Ровно та же логика, что в журнале
    эволюции (evolution_log.load_records).
    """
    meta_path = candidate_dir / META_FILE
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{meta_path} не разбирается как JSON: {e}") from e


def _write_meta(candidate_dir: Path, meta: dict) -> None:
    (candidate_dir / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=JSON_INDENT), encoding="utf-8")


def _generation_of(path: Path) -> int:
    """Поколение родителя; кандидат без meta.json считается корнем линии."""
    return int(_read_meta(path).get("generation", INITIAL_GENERATION))


def _check_patch(patch: CandidatePatch) -> None:
    """Проверки, которые можно сделать до любой записи на диск."""
    for role, text in patch.prompts.items():
        if role not in ROLES:
            raise ValueError(f"патч промпта для неизвестной роли: {role}")
        if not text.strip():
            raise ValueError(f"пустой промпт роли {role}")


def _apply_patch(target: Path, parent_dir: Path, patch: CandidatePatch) -> None:
    """Правит уже скопированную папку потомка: конфиг, промпты, meta.json."""
    config_path = target / CONFIG_FILE
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    merged = _deep_merge(config, patch.config)
    config_path.write_text(
        yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    for role, text in patch.prompts.items():
        (target / PROMPTS_DIR / f"{role}.md").write_text(text, encoding="utf-8")
    _write_meta(target, {
        "parent": parent_dir.name,
        "generation": _generation_of(parent_dir) + 1,
        "status": PENDING_STATUS,
        "rationale": patch.rationale,
        "eval": None,
    })


def write_candidate(*, parent_dir: Path, root: Path, candidate_id: str,
                    patch: CandidatePatch) -> Path:
    """Пишет потомка родителя в root/candidate_id и возвращает путь к нему."""
    target = Path(root) / candidate_id
    if target.exists():
        raise ValueError(f"кандидат {candidate_id} уже существует")
    _check_patch(patch)

    shutil.copytree(parent_dir, target)
    try:
        _apply_patch(target, parent_dir, patch)
        load_candidate(target)      # валидация: мутант обязан грузиться
    except Exception:
        # полуготовый кандидат хуже отсутствующего: следующее поколение
        # подхватит его как настоящего и тихо испортит сравнение
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def set_status(candidate_dir: Path, status: str,
               eval_data: dict | None = None) -> None:
    """Отмечает исход оценки кандидата, не трогая остальные поля meta.json."""
    meta = _read_meta(candidate_dir)
    meta["status"] = status
    if eval_data is not None:
        meta["eval"] = eval_data
    _write_meta(candidate_dir, meta)


def ancestry(candidate_dir: Path) -> list[str]:
    """Цепочка предков от текущего кандидата к корню, по именам папок.

    Испорченный meta.json может ссылаться на самого себя или на потомка;
    поэтому обход помнит уже пройденные папки и обрывается на повторе,
    а не крутится вечно.
    """
    chain: list[str] = []
    seen: set[Path] = set()
    current: Path | None = Path(candidate_dir)
    while current is not None and current.exists():
        resolved = current.resolve()
        if resolved in seen:
            break
        seen.add(resolved)
        chain.append(current.name)
        parent = _read_meta(current).get("parent")
        current = current.parent / parent if parent else None
    return chain
