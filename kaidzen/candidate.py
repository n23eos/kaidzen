"""Кандидат = папка: config.yaml + prompts/{analyzer,researcher,refiner,judge}.md."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from kaidzen.backends.registry import (KNOWN_TYPES,
                                       TYPE_OPENAI_COMPAT,
                                       openai_compat_problem,
                                       supports_web_search)

ROLES = ("analyzer", "researcher", "refiner", "judge")
# файл-указатель на кандидата-чемпиона домена: candidates/CHAMPION-<domain>.
# Живёт здесь, а не в CLI: указатель читает run, а пишет Gate мета-лупа.
CHAMPION_PREFIX = "CHAMPION-"
# отдельно от ROLES: reporter не имеет своего файла промпта (см. load_candidate),
# но бэкенд и модель для него обязаны быть заданы — иначе executive summary
# падает только в самом конце оплаченного прогона (__main__._generate_summary)
REPORTER_ROLE = "reporter"
REQUIRED_ROLES = ROLES + (REPORTER_ROLE,)
# роль, которой веб-поиск не опция, а смысл существования
SEARCHING_ROLE = "researcher"
BACKEND_TYPE_KEY = "type"
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


class RoleConfig(BaseModel):
    """Роль знает свой транспорт и свою модель: бэкенд — имя из секции backends."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    model: str

    @field_validator("backend", "model")
    @classmethod
    def check_not_blank(cls, v: str, info) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} не должен быть пустым")
        return v


class CandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    rubric: dict[str, str]
    researcher_focus: str = ""
    analyzer_hints: str = ""
    loop: LoopConfig = Field(default_factory=LoopConfig)
    backends: dict[str, dict]
    roles: dict[str, RoleConfig]

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

    @field_validator("backends")
    @classmethod
    def check_backends(cls, v: dict[str, dict]) -> dict[str, dict]:
        """Типы бэкендов проверяем здесь, а не при сборке: опечатка в type
        должна ронять загрузку кандидата, а не первый вызов модели."""
        if not v:
            raise ValueError("backends: не объявлено ни одного бэкенда")
        unknown = {name: spec.get(BACKEND_TYPE_KEY) for name, spec in v.items()
                   if spec.get(BACKEND_TYPE_KEY) not in KNOWN_TYPES}
        if unknown:
            raise ValueError(
                f"backends: неизвестный type у бэкендов {unknown}, "
                f"допустимы {', '.join(KNOWN_TYPES)}")
        cls._check_provider_addresses(v)
        return v

    @staticmethod
    def _check_provider_addresses(backends: dict[str, dict]) -> None:
        """Адрес провайдера — той же функцией, что и у сборщика бэкендов.

        Две точки проверки (конфиг и сборка) на одной реализации: разъехаться
        им не с чем, а падать конфиг обязан раньше — на загрузке кандидата.
        """
        problems = [openai_compat_problem(name, spec)
                    for name, spec in backends.items()
                    if spec.get(BACKEND_TYPE_KEY) == TYPE_OPENAI_COMPAT]
        found = [p for p in problems if p]
        if found:
            raise ValueError("; ".join(found))

    @model_validator(mode="after")
    def check_roles(self) -> "CandidateConfig":
        """Расстановка ролей по бэкендам — вся проверка до первого вызова."""
        self._check_roles_complete()
        self._check_backends_declared()
        self._check_researcher_can_search()
        return self

    def _check_roles_complete(self) -> None:
        missing = [r for r in REQUIRED_ROLES if r not in self.roles]
        if missing:
            raise ValueError(f"roles: не заданы роли {missing}")
        unexpected = [r for r in self.roles if r not in REQUIRED_ROLES]
        if unexpected:
            raise ValueError(f"roles: неизвестные роли {unexpected}")

    def _check_backends_declared(self) -> None:
        unknown = {role: cfg.backend for role, cfg in self.roles.items()
                   if cfg.backend not in self.backends}
        if unknown:
            raise ValueError(
                f"roles: бэкенды {unknown} не объявлены в секции backends "
                f"(есть: {', '.join(self.backends)})")

    def _check_researcher_can_search(self) -> None:
        """Researcher без веб-поиска = полировка идеи в вакууме (ТЗ §3.3).

        Способность спрашивается у слоя бэкендов по типу, а не по имени:
        конфиг волен назвать бэкенд как угодно.
        """
        name = self.roles[SEARCHING_ROLE].backend
        backend_type = self.backends[name].get(BACKEND_TYPE_KEY)
        if supports_web_search(backend_type):
            return
        raise ValueError(
            f"роль {SEARCHING_ROLE} стоит на бэкенде '{name}' "
            f"(type={backend_type}), который не умеет веб-поиск: без поиска "
            f"Researcher не может закрывать допущения фактами и вернёт "
            f"ссылки из памяти модели. Поставьте роль на бэкенд с поиском.")


class Candidate(BaseModel):
    candidate_id: str
    config: CandidateConfig
    prompts: dict[str, str]


def backends_by_role(candidate: "Candidate",
                     built: dict[str, object]) -> dict[str, object]:
    """Роль → её экземпляр бэкенда. Один бэкенд может обслуживать несколько ролей.

    Дальше по коду цикл видит только это отображение и не знает ни имён
    бэкендов, ни их типов: транспорт для роли выбран один раз, на старте.
    """
    return {role: built[cfg.backend]
            for role, cfg in candidate.config.roles.items()}


def load_candidate(path: Path) -> Candidate:
    config_path = path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"нет config.yaml в {path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        # main() ловит ValueError и FileNotFoundError; YAMLError мимо этого
        # списка проходил трейсбеком, хотя опечатка в отступе — обычная
        # ошибка пользователя, а не сбой программы
        raise ValueError(f"{config_path} не разбирается как YAML: {e}") from e
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
