import pytest

from kaidzen.roles.analyzer import AnalyzerOutput, run_analyzer
from kaidzen.roles.analyzer import EFFORT as analyzer_effort
from kaidzen.roles.researcher import ResearchFinding, ResearcherOutput, run_researcher
from kaidzen.roles.researcher import EFFORT as researcher_effort
from kaidzen.roles.refiner import RefinerOutput, run_refiner
from kaidzen.roles.refiner import EFFORT as refiner_effort, MAX_TOKENS as refiner_max_tokens
from kaidzen.roles.judge import run_judge
from kaidzen.roles.judge import EFFORT as judge_effort
from kaidzen.state import Assumption, ChangelogEntry, Fact, JudgeResult
from tests.conftest import FakeLLM


def _analyzer_out():
    return AnalyzerOutput(
        problem="p", audience="a", mechanism="m", unknowns=["u"],
        assumptions=[Assumption(id="A1", text="t", criticality="high")])


def test_analyzer_passes_idea_domain_and_hints(candidate):
    llm = FakeLLM([_analyzer_out()])
    out = run_analyzer(llm, candidate, idea_text="моя идея")
    assert out.assumptions[0].id == "A1"
    call = llm.calls[0]
    assert "моя идея" in call["user"]
    assert candidate.config.domain in call["user"]
    assert call["system"] == candidate.prompts["analyzer"]
    assert call["model"] == candidate.config.models["analyzer"]
    assert call["effort"] == analyzer_effort
    assert call["schema"] is AnalyzerOutput


def test_researcher_uses_web_search_and_lists_assumptions(candidate):
    finding = ResearchFinding(assumption_id="A1", verdict="confirmed",
                              facts=[Fact(claim="c", source_url="http://x")])
    llm = FakeLLM([ResearcherOutput(findings=[finding])])
    out = run_researcher(llm, candidate, idea_text="i", assumptions=[
        Assumption(id="A1", text="рынок есть", criticality="high"),
        Assumption(id="A2", text="юзеры платят", criticality="medium")])
    assert out.findings[0].verdict == "confirmed"
    call = llm.calls[0]
    assert call["web_search"] is True
    assert call["effort"] == researcher_effort
    assert "A1" in call["user"] and "рынок есть" in call["user"]
    assert "A2" in call["user"] and "юзеры платят" in call["user"]
    assert candidate.config.researcher_focus in call["user"]


def test_refiner_gets_findings_and_critique(candidate):
    llm = FakeLLM([RefinerOutput(idea_text="v2", changelog=[
        ChangelogEntry(change="c", reason="r", grounded_in=["A1"])])])
    out = run_refiner(llm, candidate, idea_text="v1",
                      findings_json='{"findings": []}', critique=["слабое место"])
    assert out.idea_text == "v2"
    call = llm.calls[0]
    assert "v1" in call["user"]
    assert "слабое место" in call["user"]
    assert '{"findings": []}' in call["user"]
    assert call["effort"] == refiner_effort


def test_refiner_handles_empty_critique(candidate):
    llm = FakeLLM([RefinerOutput(idea_text="v2")])
    run_refiner(llm, candidate, idea_text="v1", findings_json="{}", critique=[])
    assert "критики нет" in llm.calls[0]["user"]


def test_judge_gets_rubric_registry_and_both_versions(candidate):
    jr = JudgeResult(scores={k: 5.0 for k in candidate.config.rubric}, total=25.0,
                     delta_vs_previous=0.0, critique=[], verdict="continue")
    llm = FakeLLM([jr])
    run_judge(llm, candidate, new_idea="НОВЫЙ ТЕКСТ", previous_idea="СТАРЫЙ ТЕКСТ",
              assumptions=[Assumption(id="A1", text="рынок есть",
                                      criticality="high", status="confirmed")])
    call = llm.calls[0]
    for axis in candidate.config.rubric:
        assert axis in call["user"]
    assert "НОВЫЙ ТЕКСТ" in call["user"] and "СТАРЫЙ ТЕКСТ" in call["user"]
    assert "A1" in call["user"] and "confirmed" in call["user"]
    assert call["effort"] == judge_effort
    assert call["schema"] is JudgeResult


def test_researcher_names_exact_assumption_ids_to_return(candidate):
    """Свой формат id — и оркестратор не сматчит ни одного факта."""
    llm = FakeLLM([ResearcherOutput(findings=[])])
    run_researcher(llm, candidate, idea_text="i", assumptions=[
        Assumption(id="A1", text="рынок есть", criticality="high"),
        Assumption(id="A7", text="юзеры платят", criticality="medium")])
    user = llm.calls[0]["user"]
    assert "верни ровно эти assumption_id: A1, A7" in user


def test_refiner_gets_extra_token_headroom(candidate):
    """Новая версия идеи плюс changelog не должны обрезаться по max_tokens."""
    llm = FakeLLM([RefinerOutput(idea_text="v2")])
    run_refiner(llm, candidate, idea_text="v1", findings_json="{}", critique=[])
    assert llm.calls[0]["max_tokens"] == refiner_max_tokens
    assert refiner_max_tokens > 16000


def test_judge_names_exact_rubric_keys_to_return(candidate):
    jr = JudgeResult(scores={k: 5.0 for k in candidate.config.rubric}, total=25.0,
                     delta_vs_previous=0.0, critique=[], verdict="continue")
    llm = FakeLLM([jr])
    run_judge(llm, candidate, new_idea="v2", previous_idea="v1", assumptions=[])
    user = llm.calls[0]["user"]
    expected = ", ".join(candidate.config.rubric)
    assert f"верни ровно эти ключи: {expected}" in user


def test_judge_rejects_scores_with_wrong_axes(candidate):
    """Русские или самовыдуманные оси ломают рубрику молча — ловим явно."""
    jr = JudgeResult(scores={"новизна": 5.0, "ясность": 3.0}, total=8.0,
                     delta_vs_previous=0.0, critique=[], verdict="continue")
    llm = FakeLLM([jr])
    with pytest.raises(ValueError) as exc:
        run_judge(llm, candidate, new_idea="v2", previous_idea="v1", assumptions=[])
    message = str(exc.value)
    assert "groundedness" in message   # пропущенная ось названа
    assert "новизна" in message        # лишняя ось названа


def test_judge_never_sees_changelog(candidate):
    """Judge должен оценивать результат, а не рассказ об изменениях."""
    jr = JudgeResult(scores={k: 5.0 for k in candidate.config.rubric}, total=25.0,
                     delta_vs_previous=0.0, critique=[], verdict="continue")
    llm = FakeLLM([jr])
    run_judge(llm, candidate, new_idea="v2", previous_idea="v1",
              assumptions=[Assumption(id="A1", text="t", criticality="high")])
    payload = str(llm.calls[0]).lower()
    assert "changelog" not in payload
    assert "grounded_in" not in payload
