"""Тесты CLI: чистые хелперы, парсер и оффлайн-команды. Сеть не трогаем."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaidzen import __main__ as cli
from kaidzen.state import (ApiUsage, Assumption, JudgeResult, RunState, Version,
                           load_state)


class ExplodingLLM:
    """Заглушка вместо LLMClient: падает при попытке создать клиента."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("LLM-клиент не должен создаваться в этой команде")


def make_champion(root: Path, domain: str, candidate_name: str,
                  *, create_dir: bool = True) -> None:
    """Кладёт файл-указатель CHAMPION-<domain> и (опционально) сам каталог."""
    root.mkdir(parents=True, exist_ok=True)
    (root / f"CHAMPION-{domain}").write_text(candidate_name + "\n",
                                             encoding="utf-8")
    if create_dir:
        (root / candidate_name).mkdir(parents=True, exist_ok=True)


def make_state_file(run_dir: Path, **overrides) -> Path:
    """Минимальный валидный state.json в run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_dir.name,
        "candidate_id": "gen000-generic",
        "config": {},
        "original_idea": "Сырая идея.",
        "assumptions": [],
        "versions": [],
        "iteration": 0,
    }
    state.update(overrides)
    path = run_dir / "state.json"
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return path


# --- resolve_candidate_dir -------------------------------------------------

def test_resolve_candidate_dir_returns_champion_directory(tmp_path):
    # Arrange
    make_champion(tmp_path, "generic", "gen000-generic")

    # Act
    result = cli.resolve_candidate_dir(candidates_root=tmp_path,
                                       domain="generic")

    # Assert
    assert result == tmp_path / "gen000-generic"


def test_resolve_candidate_dir_raises_when_pointer_missing(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        cli.resolve_candidate_dir(candidates_root=tmp_path, domain="games")

    assert "CHAMPION-games" in str(exc.value)


def test_resolve_candidate_dir_raises_when_pointed_directory_missing(tmp_path):
    make_champion(tmp_path, "business", "gen999-business", create_dir=False)

    with pytest.raises(FileNotFoundError) as exc:
        cli.resolve_candidate_dir(candidates_root=tmp_path, domain="business")

    assert "gen999-business" in str(exc.value)


def test_resolve_candidate_dir_raises_when_pointer_empty(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "CHAMPION-generic").write_text("  \n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        cli.resolve_candidate_dir(candidates_root=tmp_path, domain="generic")

    assert "CHAMPION-generic" in str(exc.value)


# --- make_run_dir ----------------------------------------------------------

@pytest.mark.parametrize("stem, expected_slug", [
    ("My Idea", "my-idea"),
    ("idea!!!  v2", "idea-v2"),
    ("---edge---", "edge"),
    ("Моя Идея", "моя-идея"),
    ("!!!", "idea"),
])
def test_make_run_dir_slugifies_stem(tmp_path, stem, expected_slug):
    # Arrange
    idea_path = tmp_path / f"{stem}.md"
    now_str = "2026-08-03-1215"

    # Act
    run_dir = cli.make_run_dir(runs_root=tmp_path / "runs",
                               idea_path=idea_path, now_str=now_str)

    # Assert
    assert run_dir == tmp_path / "runs" / f"{now_str}-{expected_slug}"


def test_make_run_dir_does_not_create_directory(tmp_path):
    run_dir = cli.make_run_dir(runs_root=tmp_path / "runs",
                               idea_path=tmp_path / "idea.md",
                               now_str="2026-08-03-1215")

    assert not run_dir.exists()


# --- парсер ----------------------------------------------------------------

def test_parser_accepts_run_with_all_arguments():
    args = cli.build_parser().parse_args(
        ["run", "idea.md", "--domain", "games", "--candidate", "candidates/x",
         "--max-iter", "3"])

    assert (args.command, args.idea, args.domain) == ("run", "idea.md", "games")
    assert (args.candidate, args.max_iter) == ("candidates/x", 3)


def test_parser_run_defaults_to_generic_domain():
    args = cli.build_parser().parse_args(["run", "idea.md"])

    assert args.domain == "generic"
    assert args.candidate is None and args.max_iter is None


@pytest.mark.parametrize("command", ["resume", "report"])
def test_parser_accepts_run_dir_commands(command):
    args = cli.build_parser().parse_args([command, "runs/2026-08-03-1215-idea"])

    assert args.command == command
    assert args.run_dir == "runs/2026-08-03-1215-idea"


def test_parser_rejects_unknown_domain():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "idea.md", "--domain", "cooking"])


# --- команда report --------------------------------------------------------

def test_report_command_writes_report_without_llm(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr(cli, "LLMClient", ExplodingLLM)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    run_dir = tmp_path / "2026-08-03-1215-idea"
    make_state_file(run_dir, stop_reason="plateau", iteration=2)

    # Act
    cli.main(["report", str(run_dir)])

    # Assert
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "2026-08-03-1215-idea" in report
    assert "Сырая идея." in report
    assert str(run_dir / "report.md") in capsys.readouterr().out


def test_report_command_fails_clearly_without_state(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["report", str(tmp_path)])

    assert "state.json" in str(exc.value)


# --- команда resume --------------------------------------------------------

def test_resume_refuses_finished_run(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setattr(cli, "LLMClient", ExplodingLLM)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    run_dir = tmp_path / "2026-08-03-1215-idea"
    make_state_file(run_dir, stop_reason="max_iterations")

    # Act
    with pytest.raises(SystemExit) as exc:
        cli.main(["resume", str(run_dir)])

    # Assert
    message = str(exc.value)
    assert message and message != "0"
    assert "report" in message


def test_resume_fails_clearly_without_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with pytest.raises(SystemExit) as exc:
        cli.main(["resume", str(tmp_path)])

    assert "state.json" in str(exc.value)


def test_resume_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    run_dir = tmp_path / "2026-08-03-1215-idea"
    make_state_file(run_dir)

    with pytest.raises(SystemExit) as exc:
        cli.main(["resume", str(run_dir)])

    assert "ANTHROPIC_API_KEY" in str(exc.value)


# --- команда run -----------------------------------------------------------

def test_run_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", str(tmp_path / "idea.md")])

    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_run_fails_clearly_when_idea_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "LLMClient", ExplodingLLM)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    missing = tmp_path / "нет-такого.md"

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", str(missing)])

    assert str(missing) in str(exc.value)


# --- вывод хода и завершение прогона --------------------------------------

class FakeSummaryLLM:
    """Отдаёт готовое резюме и фиктивный расход. В сеть не ходит."""

    def __init__(self, summary: str):
        self.usage = ApiUsage(input_tokens=11, output_tokens=22, web_searches=3)
        self._summary = summary

    def structured(self, **kwargs):
        return cli.SummaryOutput(summary=self._summary)


def make_judge(total: float, delta: float) -> JudgeResult:
    return JudgeResult(scores={"groundedness": total}, total=total,
                       delta_vs_previous=delta, critique=[], verdict="continue")


def test_progress_printer_analyzer_reports_count_and_high_criticality(capsys):
    state = RunState(run_id="r", candidate_id="c", config={}, original_idea="и",
                     assumptions=[
                         Assumption(id="A1", text="x", criticality="high"),
                         Assumption(id="A2", text="y", criticality="low"),
                     ])

    cli.ProgressPrinter()(cli.STEP_ANALYZER, state)

    out = capsys.readouterr().out
    assert "допущений найдено: 2" in out
    assert "критичных: 1" in out


def test_progress_printer_researcher_names_only_changed_ids(capsys):
    # Arrange: принтер видел допущения через analyzer, все unverified
    printer = cli.ProgressPrinter()
    state = RunState(run_id="r", candidate_id="c", config={}, original_idea="и",
                     assumptions=[
                         Assumption(id="A1", text="x", criticality="high"),
                         Assumption(id="A2", text="y", criticality="low"),
                     ])
    printer(cli.STEP_ANALYZER, state)
    capsys.readouterr()  # сбрасываем вывод шага analyzer

    # Act: только A1 сменил статус, A2 остался unverified
    state.assumptions[0].status = "confirmed"
    printer(cli.STEP_RESEARCHER, state)

    # Assert
    out = capsys.readouterr().out
    assert "A1=confirmed" in out
    assert "A2" not in out


def test_progress_printer_refiner_reports_new_version_number(capsys):
    state = RunState(run_id="r", candidate_id="c", config={}, original_idea="и",
                     versions=[Version(n=1, idea_text="v1")])

    cli.ProgressPrinter()(cli.STEP_REFINER, state)

    assert "v1" in capsys.readouterr().out


def test_progress_printer_judge_reports_iteration_score_and_delta(capsys):
    state = RunState(run_id="r", candidate_id="c", config={}, original_idea="и",
                     iteration=1,
                     versions=[Version(n=1, idea_text="v1",
                                       judge=make_judge(7.0, 2.0))])

    cli.ProgressPrinter()(cli.STEP_JUDGE, state)

    out = capsys.readouterr().out
    assert "итерация 1" in out and "7.0" in out and "+2.0" in out


def test_progress_printer_judge_reports_rollback(capsys):
    state = RunState(run_id="r", candidate_id="c", config={}, original_idea="и",
                     iteration=1,
                     versions=[Version(n=1, idea_text="v1", rolled_back=True)])

    cli.ProgressPrinter()(cli.STEP_JUDGE, state)

    assert "откачена" in capsys.readouterr().out


def test_finish_run_writes_report_with_summary_and_prints_usage(tmp_path, capsys):
    # Arrange
    run_dir = tmp_path / "2026-08-03-1215-idea"
    run_dir.mkdir()
    state = RunState(run_id=run_dir.name, candidate_id="gen000-generic",
                     config={}, original_idea="Сырая идея.",
                     stop_reason="plateau", iteration=1,
                     versions=[Version(n=1, idea_text="Доведённая идея.",
                                       judge=make_judge(8.0, 1.5))])
    llm = FakeSummaryLLM("Короткое резюме идеи.")

    # Act
    cli._finish_run(llm, state, run_dir)

    # Assert
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Короткое резюме идеи." in report
    assert "Доведённая идея." in report
    out = capsys.readouterr().out
    assert "Остановка: plateau" in out
    assert "11 входных, 22 выходных" in out and "веб-поисков: 3" in out
    # расход на резюме должен осесть и в state.json, а не только в отчёте
    assert load_state(run_dir).api_usage.input_tokens == 11


def test_print_start_shows_run_and_candidate(candidate, capsys):
    cli._print_start(Path("runs/2026-08-03-1215-idea"), candidate)

    out = capsys.readouterr().out
    assert "2026-08-03-1215-idea" in out
    assert candidate.candidate_id in out


def test_apply_max_iter_returns_new_candidate_without_mutating(candidate):
    # Arrange
    original = candidate.config.loop.max_iterations

    # Act
    updated = cli.apply_max_iter(candidate, original + 1)

    # Assert
    assert updated.config.loop.max_iterations == original + 1
    assert candidate.config.loop.max_iterations == original


def test_apply_max_iter_keeps_candidate_when_not_given(candidate):
    assert cli.apply_max_iter(candidate, None) is candidate
