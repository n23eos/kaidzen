"""Схемы состояния прогона и атомарная запись state.json."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Criticality = Literal["high", "medium", "low"]
AssumptionStatus = Literal["unverified", "confirmed", "refuted", "partial", "untestable"]


class Fact(BaseModel):
    claim: str
    source_url: str
    source_title: str = ""
    date: str = ""


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
    """Атомарно: пишем во временный файл, затем rename поверх старого."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "state.json")


def load_state(run_dir: Path) -> RunState:
    path = run_dir / "state.json"
    if not path.exists():
        raise FileNotFoundError(f"state.json не найден в {run_dir}")
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))
