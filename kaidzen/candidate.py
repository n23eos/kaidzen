"""Кандидат = папка: config.yaml + prompts/{analyzer,researcher,refiner,judge}.md."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROLES = ("analyzer", "researcher", "refiner", "judge")
RUBRIC_AXES = 5

# границы параметров цикла: за ними прогон либо не стартует,
# либо никогда не останавливается по плато
MIN_ITERATIONS, MAX_ITERATIONS = 1, 20
MIN_PLATEAU_THRESHOLD, MAX_PLATEAU_THRESHOLD = 0.0, 10.0
MIN_ASSUMPTIONS_PER_ITERATION, MAX_ASSUMPTIONS_PER_ITERATION = 1, 10


class LoopConfig(BaseModel):
    # extra="forbid": опечатка в ключе не должна молча включать дефолт
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=6, ge=MIN_ITERATIONS, le=MAX_ITERATIONS)
    plateau_threshold: float = Field(default=0.5, ge=MIN_PLATEAU_THRESHOLD,
                                     le=MAX_PLATEAU_THRESHOLD)
    assumptions_per_iteration: int = Field(
        default=3, ge=MIN_ASSUMPTIONS_PER_ITERATION,
        le=MAX_ASSUMPTIONS_PER_ITERATION)


class CandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    rubric: dict[str, str]
    researcher_focus: str = ""
    analyzer_hints: str = ""
    loop: LoopConfig = Field(default_factory=LoopConfig)
    models: dict[str, str]

    @field_validator("domain")
    @classmethod
    def check_domain(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("domain не должен быть пустым")
        return v

    @field_validator("rubric")
    @classmethod
    def check_rubric(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) != RUBRIC_AXES:
            raise ValueError(
                f"рубрика должна содержать ровно {RUBRIC_AXES} осей, получено {len(v)}")
        if "groundedness" not in v:
            raise ValueError("ось groundedness обязательна в рубрике")
        blank = [axis for axis, desc in v.items() if not desc.strip()]
        if blank:
            raise ValueError(f"рубрика: пустое описание осей {blank}")
        return v

    @field_validator("models")
    @classmethod
    def check_models(cls, v: dict[str, str]) -> dict[str, str]:
        missing = [r for r in ROLES if r not in v]
        if missing:
            raise ValueError(f"models: не заданы модели для ролей {missing}")
        blank = [r for r in ROLES if not v[r].strip()]
        if blank:
            raise ValueError(f"models: пустое имя модели для ролей {blank}")
        return v


class Candidate(BaseModel):
    candidate_id: str
    config: CandidateConfig
    prompts: dict[str, str]


def load_candidate(path: Path) -> Candidate:
    config_path = path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"нет config.yaml в {path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        # пустой файл даёт None, список верхнего уровня — list;
        # без этой проверки пользователь получит невнятную ошибку pydantic
        raise ValueError(
            f"config.yaml должен быть словарём, получено {type(data).__name__}")
    config = CandidateConfig.model_validate(data)
    prompts: dict[str, str] = {}
    for role in ROLES:
        p = path / "prompts" / f"{role}.md"
        if not p.exists():
            raise FileNotFoundError(f"нет промпта роли {role}: {p}")
        text = p.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"промпт роли {role} пустой: {p}")
        prompts[role] = text
    return Candidate(candidate_id=path.name, config=config, prompts=prompts)
