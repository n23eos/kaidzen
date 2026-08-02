"""Тесты сборщика отчёта: build_report собирает markdown детерминированно из RunState."""
import pytest

from kaidzen.report import build_report
from kaidzen.state import (ApiUsage, Assumption, ChangelogEntry, Fact,
                            JudgeResult, RunState, Version)


def _judge(scores: dict, delta: float = 0.0) -> JudgeResult:
    return JudgeResult(scores=scores, total=sum(scores.values()),
                        delta_vs_previous=delta, critique=[], verdict="continue")


def _full_state() -> RunState:
    """Полноценный завершённый прогон с двумя оценёнными версиями и допущениями."""
    assumptions = [
        Assumption(
            id="A1", text="рынок существует", criticality="high",
            status="confirmed",
            facts=[Fact(claim="рынок растёт на 10% в год",
                        source_url="https://example.com/report",
                        source_title="Market Report 2026")],
        ),
        Assumption(
            id="A2", text="конкурентов нет", criticality="medium",
            status="untestable", facts=[],
        ),
        Assumption(
            id="A3", text="цена приемлема | со скидкой", criticality="low",
            status="unverified",
            facts=[Fact(claim="цена конкурентна",
                        source_url="https://example.com/price")],
        ),
    ]
    versions = [
        Version(n=1, idea_text="идея v1",
                changelog=[ChangelogEntry(change="уточнили аудиторию",
                                           reason="анализ показал разброс")],
                judge=_judge({"clarity": 5.0, "novelty": 3.0})),
        Version(n=2, idea_text="идея v2",
                changelog=[ChangelogEntry(change="добавили механику",
                                           reason="фидбек судьи"),
                           ChangelogEntry(change="убрали лишнее",
                                          reason="упрощение"),
                           ChangelogEntry(change="третье изменение",
                                          reason="не должно попасть в отчёт")],
                judge=_judge({"clarity": 8.0, "novelty": 6.0}, delta=6.0)),
    ]
    return RunState(
        run_id="run-42", candidate_id="gen000-generic", config={},
        original_idea="сырая идея", assumptions=assumptions, versions=versions,
        iteration=2, stop_reason="plateau",
        api_usage=ApiUsage(input_tokens=1000, output_tokens=500, web_searches=7),
    )


def test_header_and_summary_present():
    state = _full_state()
    report = build_report(state, summary_text="Идея готова к пилоту.")
    assert "run-42" in report
    assert "gen000-generic" in report
    assert "plateau" in report
    assert "Идея готова к пилоту." in report


def test_final_idea_section_present():
    state = _full_state()
    report = build_report(state)
    assert "Финальная версия идеи" in report
    assert "идея v2" in report


def test_rubric_table_shows_first_and_final_scored_versions():
    state = _full_state()
    report = build_report(state)
    assert "Оценки по рубрике" in report
    assert "clarity" in report and "novelty" in report
    # первая версия: clarity=5.0, финальная: clarity=8.0
    assert "5.0" in report and "8.0" in report
    assert "total" in report.lower()


def test_assumptions_table_present():
    state = _full_state()
    report = build_report(state)
    assert "Допущения" in report
    assert "A1" in report and "рынок существует" in report
    assert "high" in report and "confirmed" in report


def test_fact_link_renders_with_title():
    state = _full_state()
    report = build_report(state)
    assert "[Market Report 2026](https://example.com/report): рынок растёт на 10% в год" in report


def test_fact_without_title_falls_back_to_url_as_link_text():
    state = _full_state()
    report = build_report(state)
    assert "[https://example.com/price](https://example.com/price): цена конкурентна" in report


def test_assumption_without_facts_renders_dash():
    state = _full_state()
    report = build_report(state)
    lines = [l for l in report.splitlines() if l.startswith("| A2")]
    assert len(lines) == 1
    assert "| - |" in lines[0] or lines[0].rstrip().endswith("| - |")


def test_untestable_assumptions_appear_in_next_steps():
    state = _full_state()
    report = build_report(state)
    next_steps = report.split("Next steps")[1]
    assert "конкурентов нет" in next_steps


def test_confirmed_assumptions_do_not_appear_in_next_steps():
    state = _full_state()
    report = build_report(state)
    next_steps = report.split("Next steps")[1]
    assert "рынок существует" not in next_steps


def test_all_checked_wording_when_nothing_untestable():
    state = _full_state()
    # заменяем untestable на confirmed
    checked = state.model_copy(update={
        "assumptions": [a.model_copy(update={"status": "confirmed"}) if a.id == "A2" else a
                        for a in state.assumptions]
    })
    report = build_report(checked)
    next_steps = report.split("Next steps")[1]
    assert "провер" in next_steps.lower()  # "все допущения проверены..."


def test_rolled_back_version_marked_and_excluded_from_rubric():
    state = _full_state()
    rolled = state.model_copy(update={
        "versions": state.versions + [
            Version(n=3, idea_text="откаченная идея", rolled_back=True,
                    judge=_judge({"clarity": 1.0, "novelty": 1.0}))
        ]
    })
    report = build_report(rolled)
    evolution = report.split("Эволюция версий")[1].split("Next steps")[0]
    assert "откачен" in evolution.lower() or "rolled" in evolution.lower()
    # оценка откаченной версии (1.0/1.0) не должна попасть в таблицу рубрики
    rubric = report.split("Оценки по рубрике")[1].split("Допущения")[0]
    assert "1.0" not in rubric


def test_evolution_limits_to_first_two_changelog_entries():
    state = _full_state()
    report = build_report(state)
    evolution = report.split("Эволюция версий")[1].split("Next steps")[0]
    assert "добавили механику" in evolution
    assert "убрали лишнее" in evolution
    assert "третье изменение" not in evolution


def test_zero_scored_versions_does_not_crash():
    state = RunState(run_id="r1", candidate_id="c", config={},
                      original_idea="идея без версий")
    report = build_report(state)
    assert "Оценки по рубрике" in report
    assert "run-report-should-not-crash" not in report  # просто убеждаемся что дошли сюда


def test_pipe_character_in_cell_is_escaped_and_does_not_break_table():
    state = _full_state()
    report = build_report(state)
    # строка с "|" внутри допущения A3 не должна порождать лишний столбец
    for line in report.splitlines():
        if line.startswith("| A3"):
            assert "\\|" in line  # исходный "|" экранирован
            # разделителей столбцов (неэкранированных "|") ровно 6 для 5 полей
            unescaped_pipes = line.replace("\\|", "").count("|")
            assert unescaped_pipes == 6


def test_api_usage_numbers_present():
    state = _full_state()
    report = build_report(state)
    assert "1000" in report
    assert "500" in report
    assert "7" in report


def test_missing_summary_text_defaults_to_empty_and_no_crash():
    state = _full_state()
    report = build_report(state)  # без summary_text
    assert isinstance(report, str) and len(report) > 0
