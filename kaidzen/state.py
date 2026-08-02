"""Схемы состояния прогона и атомарная запись state.json."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Criticality = Literal["high", "medium", "low"]
AssumptionStatus = Literal["unverified", "confirmed", "refuted", "partial", "untestable"]

ALLOWED_URL_SCHEMES = ("http://", "https://")
MAX_AXIS_SCORE = 10.0
MIN_AXIS_SCORE = 0.0
TOTAL_TOLERANCE = 0.01


class Fact(BaseModel):
    claim: str
    source_url: str
    source_title: str = ""
    date: str = ""

    @field_validator("source_url")
    @classmethod
    def check_source_url(cls, v: str) -> str:
        # вся ось groundedness держится на этом поле: выдуманная ссылка
        # должна валить валидацию, а не попадать в отчёт
        url = v.strip()
        if not url.startswith(ALLOWED_URL_SCHEMES):
            raise ValueError(
                f"source_url должен быть http(s)-ссылкой, получено: {v!r}")
        return url


class Assumption(BaseModel):
    id: str
    text: str
    criticality: Criticality
    status: AssumptionStatus = "unverified"
    facts: list[Fact] = Field(default_factory=list)


class Analysis(BaseModel):
    problem: str
    audience: str
    mechanism: str
    unknowns: list[str] = Field(default_factory=list)


class ChangelogEntry(BaseModel):
    change: str
    reason: str
    grounded_in: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    scores: dict[str, float]
    total: float
    delta_vs_previous: float
    critique: list[str]
    verdict: Literal["continue", "rollback"]

    @model_validator(mode="after")
    def check_scores(self) -> "JudgeResult":
        # чужая шкала (например 0–100) молча ломает delta_vs_previous,
        # а на нём держится условие остановки по плато
        for axis, score in self.scores.items():
            if not MIN_AXIS_SCORE <= score <= MAX_AXIS_SCORE:
                raise ValueError(
                    f"ось {axis}: оценка {score} вне диапазона "
                    f"{MIN_AXIS_SCORE}–{MAX_AXIS_SCORE}")
        expected = sum(self.scores.values())
        if abs(self.total - expected) > TOTAL_TOLERANCE:
            raise ValueError(
                f"total={self.total} не равен сумме оценок {expected}")
        return self


class Version(BaseModel):
    n: int
    idea_text: str
    changelog: list[ChangelogEntry] = Field(default_factory=list)
    judge: Optional[JudgeResult] = None
    rolled_back: bool = False


class ApiUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0


class RunState(BaseModel):
    run_id: str
    candidate_id: str
    config: dict[str, Any]
    original_idea: str
    analysis: Optional[Analysis] = None
    assumptions: list[Assumption] = Field(default_factory=list)
    versions: list[Version] = Field(default_factory=list)
    iteration: int = 0
    rollbacks: int = 0
    consecutive_rollbacks: int = 0
    low_delta_streak: int = 0
    stop_reason: Optional[str] = None
    last_completed_step: Optional[str] = None
    api_usage: ApiUsage = Field(default_factory=ApiUsage)

    def current_version(self) -> Optional[Version]:
        """Последняя не откаченная версия идеи."""
        active = [v for v in self.versions if not v.rolled_back]
        return active[-1] if active else None

    def current_idea_text(self) -> str:
        v = self.current_version()
        return v.idea_text if v else self.original_idea


def save_state(state: RunState, run_dir: Path) -> None:
    """Атомарно и с защитой от потери питания: fsync данных, rename, fsync каталога.

    Без fsync файла rename может лечь на диск раньше самих данных — после
    отключения питания останется обрезанный state.json, а именно на него
    опирается возобновление прогона.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=run_dir, prefix="state.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, run_dir / "state.json")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(run_dir)


def _fsync_dir(path: Path) -> None:
    """Фиксирует сам rename. На платформах без O_DIRECTORY просто пропускаем."""
    flags = getattr(os, "O_DIRECTORY", None)
    if flags is None:
        return
    try:
        dir_fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def load_state(run_dir: Path) -> RunState:
    path = run_dir / "state.json"
    if not path.exists():
        raise FileNotFoundError(f"state.json не найден в {run_dir}")
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))
