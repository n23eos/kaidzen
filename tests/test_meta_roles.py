"""Мета-роли Уровня 2. Ключевой тест здесь — слепота судьи."""
import inspect

import pytest

from kaidzen.metrics import RunMetrics
from kaidzen.mutation import CandidatePatch, write_candidate
from kaidzen.roles.meta import MetaConfig, load_meta_prompt
from kaidzen.roles.meta.diagnostician import Diagnosis, run_diagnostician
from kaidzen.roles.meta.meta_judge import Comparison, run_meta_judge
from kaidzen.roles.meta.mutator import (MAX_ROLES_PER_MUTATION,
                                        MutationProposal, run_mutator, to_patch)
from tests.conftest import FakeLLM

META = MetaConfig()

# слова, любое из которых в вызове судьи означает утечку контекста сравнения
LEAK_WORDS = ("gen00", "челленджер", "чемпион", "диагноз", "мутац", "rationale")


def _diagnosis():
    return Diagnosis(weaknesses=["researcher хеджирует"], hypotheses=[
        "усилить формулировку про вердикт в промпте researcher"])


def test_diagnostician_gets_metrics_and_reports(candidate):
    llm = FakeLLM([_diagnosis()])
    out = run_diagnostician(llm, META,
                            metrics=RunMetrics(partial_rate=0.9),
                            reports=["отчёт один", "отчёт два"])
    assert out.hypotheses
    user = llm.calls[0]["user"]
    assert "0.9" in user or "0,9" in user
    assert "отчёт один" in user


def test_mutator_returns_patch_with_rationale(candidate):
    llm = FakeLLM([MutationProposal(
        prompts={"researcher": "новый текст"}, config={}, rationale="по диагнозу")])
    out = run_mutator(llm, META, candidate, diagnosis=_diagnosis(), attempt=0)
    assert out.rationale
    assert "researcher" in out.prompts
    assert "хеджирует" in llm.calls[0]["user"]


def test_meta_judge_sees_only_two_reports(candidate):
    llm = FakeLLM([Comparison(winner="A", reason="больше закрытых допущений")])
    run_meta_judge(llm, META, report_a="ОТЧЁТ А", report_b="ОТЧЁТ Б")
    payload = str(llm.calls[0]).lower()
    assert "отчёт а" in payload and "отчёт б" in payload
    for leak in LEAK_WORDS:
        assert leak not in payload, f"судье утекло: {leak}"


def test_meta_judge_temperature_is_deterministic(candidate):
    llm = FakeLLM([Comparison(winner="B", reason="r")])
    run_meta_judge(llm, META, report_a="a", report_b="b")
    assert llm.calls[0]["effort"] == "low"


def test_meta_judge_signature_has_no_room_for_context():
    """Слепота — свойство сигнатуры, а не дисциплины вызывающего кода.

    Проверка кажется избыточной рядом с проверкой полезной нагрузки, но она
    ловит другое: пока лишнего параметра нет, оркестратор Task 6 физически не
    сможет передать судье id кандидата «на всякий случай».
    """
    params = list(inspect.signature(run_meta_judge).parameters)
    assert params == ["backend", "meta", "report_a", "report_b"]
    # meta — это транспорт и имя модели судьи; кандидата он не видит вовсе
    assert "candidate" not in params


def test_meta_judge_prompt_itself_mentions_nothing_about_origin(candidate):
    """Утечка через системный промпт так же смертельна, как через user."""
    system = load_meta_prompt("meta_judge").lower()
    for leak in LEAK_WORDS:
        assert leak not in system, f"промпт судьи упоминает: {leak}"


def test_meta_judge_can_return_tie(candidate):
    llm = FakeLLM([Comparison(winner="tie", reason="равны по существу")])
    out = run_meta_judge(llm, META, report_a="a", report_b="b")
    assert out.winner == "tie"


def test_meta_judge_prompt_allows_tie():
    assert "tie" in load_meta_prompt("meta_judge")


def test_mutator_attempt_changes_the_prompt(candidate):
    """Два челленджера поколения обязаны получить разные задания."""
    llm = FakeLLM([MutationProposal(rationale="r"), MutationProposal(rationale="r")])
    run_mutator(llm, META, candidate, diagnosis=_diagnosis(), attempt=0)
    run_mutator(llm, META, candidate, diagnosis=_diagnosis(), attempt=1)
    assert llm.calls[0]["user"] != llm.calls[1]["user"]


def test_mutation_proposal_converts_to_valid_patch(tmp_path, candidate):
    """Предложение модели должно доезжать до диска без ручной сборки патча."""
    from tests.test_candidate import make_candidate

    parent = make_candidate(tmp_path / "parent")
    proposal = MutationProposal(
        prompts={"researcher": "требуй вердикта"},
        config={"loop": {"max_iterations": 4}}, rationale="по гипотезе 1")
    patch = to_patch(proposal)
    assert isinstance(patch, CandidatePatch)

    child = write_candidate(parent_dir=parent, root=tmp_path / "out",
                            candidate_id="gen001-child", patch=patch)
    assert (child / "prompts" / "researcher.md").read_text(
        encoding="utf-8") == "требуй вердикта"


def test_deep_roles_declare_their_effort(candidate):
    """Диагност и мутатор рассуждают, а не пересказывают — им нужен high."""
    llm = FakeLLM([_diagnosis(), MutationProposal(rationale="r")])
    run_diagnostician(llm, META, metrics=RunMetrics(), reports=["о"])
    run_mutator(llm, META, candidate, diagnosis=_diagnosis(), attempt=0)
    assert [c["effort"] for c in llm.calls] == ["high", "high"]


def test_missing_meta_prompt_is_a_clear_error():
    with pytest.raises(FileNotFoundError, match="нет промпта мета-роли"):
        load_meta_prompt("нет-такой-роли")


def test_models_come_from_meta_config_not_from_candidate(candidate):
    """Модель мета-уровня задаётся evolve-прогоном.

    Раньше она бралась у роли reporter проверяемого кандидата — то есть
    мутация конфига кандидата могла молча сменить модель судьи и вместе с ней
    планку сравнения поколений.
    """
    meta = MetaConfig(deep_model="глубокая", judge_model="судейская")
    llm = FakeLLM([_diagnosis(), MutationProposal(rationale="r"),
                   Comparison(winner="tie", reason="r")])
    run_diagnostician(llm, meta, metrics=RunMetrics(), reports=["о"])
    run_mutator(llm, meta, candidate, diagnosis=_diagnosis(), attempt=0)
    run_meta_judge(llm, meta, report_a="a", report_b="b")
    assert [c["model"] for c in llm.calls] == ["глубокая", "глубокая", "судейская"]
    reporter_model = candidate.config.roles["reporter"].model
    assert reporter_model not in [c["model"] for c in llm.calls]


def test_meta_config_defaults_to_subscription():
    """Мета-прогон по умолчанию не требует ни одного ключа."""
    assert MetaConfig().backend == {"type": "claude_agent_sdk"}


def test_patch_touching_three_roles_is_rejected():
    """Иначе непонятно, какая из правок дала выигрыш поколения."""
    proposal = MutationProposal(prompts={"analyzer": "a", "researcher": "b",
                                         "refiner": "c"})
    with pytest.raises(ValueError, match=str(MAX_ROLES_PER_MUTATION)):
        to_patch(proposal)


def test_patch_touching_two_roles_is_allowed():
    patch = to_patch(MutationProposal(prompts={"analyzer": "a", "judge": "b"}))
    assert set(patch.prompts) == {"analyzer", "judge"}
