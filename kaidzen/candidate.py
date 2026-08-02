"""Кандидат = папка: config.yaml + prompts/{analyzer,researcher,refiner,judge}.md."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

ROLES = ("analyzer", "researcher", "refiner", "judge")
RUBRIC_AXES = 5


class LoopConfig(BaseModel):
    max_iterations: int = 6
    plateau_threshold: float = 0.5
    assumptions_per_iteration: int = 3


class CandidateConfig(BaseModel):
    domain: str
    rubric: dict[str, str]
    researcher_focus: str = ""
    analyzer_hints: str = ""
    loop: LoopConfig = Field(default_factory=LoopConfig)
    models: dict[str, str]

    @field_validator("rubric")
    @classmethod
    def check_rubric(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) != RUBRIC_AXES:
            raise ValueError(
                f"рубрика должна содержать ровно {RUBRIC_AXES} осей, получено {len(v)}")
        if "groundedness" not in v:
            raise ValueError("ось groundedness обязательна в рубрике")
        return v

    @field_validator("models")
    @classmethod
    def check_models(cls, v: dict[str, str]) -> dict[str, str]:
        missing = [r for r in ROLES if r not in v]
        if missing:
            raise ValueError(f"models: не заданы модели для ролей {missing}")
        return v


class Candidate(BaseModel):
    candidate_id: str
    config: CandidateConfig
    prompts: dict[str, str]


def load_candidate(path: Path) -> Candidate:
    config_path = path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"нет config.yaml в {path}")
    config = CandidateConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8")))
    prompts: dict[str, str] = {}
    for role in ROLES:
        p = path / "prompts" / f"{role}.md"
        if not p.exists():
            raise FileNotFoundError(f"нет промпта роли {role}: {p}")
        prompts[role] = p.read_text(encoding="utf-8")
    return Candidate(candidate_id=path.name, config=config, prompts=prompts)
