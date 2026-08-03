"""Мутатор: по диагнозу правит промпты и конфиг родителя, порождая челленджера."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from kaidzen.candidate import Candidate
from kaidzen.mutation import CandidatePatch
from kaidzen.roles.meta import MetaConfig, load_meta_prompt
from kaidzen.roles.meta.diagnostician import Diagnosis

# правка чужого промпта требует понимания, зачем там каждая фраза
EFFORT = "high"
PROMPT_NAME = "mutator"
JSON_INDENT = 2
# конфиг показывается без backends и roles: транспорт и модели мутатор не трогает
EDITABLE_CONFIG_KEYS = ("loop", "rubric", "researcher_focus", "analyzer_hints")

# Правка больше двух ролей за поколение делает результат неатрибутируемым:
# кандидат выиграл, а какая из четырёх правок сработала — неизвестно, и
# следующий диагноз строится на догадке. В промпте мутатора это написано
# словами, здесь — проверено кодом: промпт модель может и не послушаться.
MAX_ROLES_PER_MUTATION = 2

# Разные углы атаки для разных челленджеров одного поколения. Без этого две
# попытки по одному диагнозу дают почти одинаковых потомков, и поколение
# сравнивает кандидата с его же копией.
ATTEMPT_ANGLES = (
    "Возьми самую сильную гипотезу и реализуй её прямолинейно: минимальная "
    "правка ровно в том месте, на которое указывает диагноз.",
    "Возьми другую гипотезу, не ту, что кажется главной, и зайди с другой "
    "стороны: если очевидная правка — ужесточить формулировку, попробуй "
    "изменить сам порядок работы роли или параметр цикла.",
)


class MutationProposal(BaseModel):
    prompts: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


def to_patch(proposal: MutationProposal) -> CandidatePatch:
    """Предложение модели → патч для write_candidate.

    Здесь же граница «не больше двух ролей за мутацию»: остальную валидацию
    (неизвестная роль, пустой промпт, невалидный конфиг) делает write_candidate,
    и дублировать её не нужно.
    """
    if len(proposal.prompts) > MAX_ROLES_PER_MUTATION:
        raise ValueError(
            f"мутация правит {len(proposal.prompts)} ролей "
            f"({', '.join(sorted(proposal.prompts))}), допустимо не больше "
            f"{MAX_ROLES_PER_MUTATION}: иначе непонятно, какая правка сработала")
    return CandidatePatch(config=proposal.config, prompts=proposal.prompts,
                          rationale=proposal.rationale)


def _editable_config(candidate: Candidate) -> str:
    data = candidate.config.model_dump()
    return json.dumps({k: data[k] for k in EDITABLE_CONFIG_KEYS},
                      ensure_ascii=False, indent=JSON_INDENT)


def _prompts_block(candidate: Candidate) -> str:
    return "\n\n".join(f"### Промпт роли {role}\n\n{text}"
                       for role, text in candidate.prompts.items())


def run_mutator(backend, meta: MetaConfig, candidate: Candidate, *,
                diagnosis: Diagnosis, attempt: int) -> MutationProposal:
    """attempt задаёт угол атаки, поэтому челленджеры поколения различаются."""
    weaknesses = "\n".join(f"- {w}" for w in diagnosis.weaknesses)
    hypotheses = "\n".join(f"- {h}" for h in diagnosis.hypotheses)
    angle = ATTEMPT_ANGLES[attempt % len(ATTEMPT_ANGLES)]
    user = (f"Слабые места родителя:\n{weaknesses}\n\n"
            f"Гипотезы улучшения:\n{hypotheses}\n\n"
            f"Правимая часть конфига родителя:\n\n{_editable_config(candidate)}\n\n"
            f"Промпты родителя:\n\n{_prompts_block(candidate)}\n\n"
            f"Это попытка №{attempt + 1}. {angle}\n\n"
            f"Верни правки не более чем для двух ролей.")
    return backend.structured(model=meta.deep_model,
                              system=load_meta_prompt(PROMPT_NAME), user=user,
                              schema=MutationProposal, effort=EFFORT)
