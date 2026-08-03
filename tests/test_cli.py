"""Тесты CLI: чистые хелперы, парсер и оффлайн-команды. Сеть не трогаем."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from kaidzen import __main__ as cli
from kaidzen.state import (ApiUsage, Assumption, JudgeResult, RunState, Version,
                           load_state)
from tests.conftest import as_backends
from tests.test_candidate import make_candidate


def explode(*args, **kwargs):
    """Вместо сборки бэкендов: команда, которой они не нужны, не должна их строить."""
    raise AssertionError("бэкенды не должны собираться в этой команде")


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
    monkeypatch.setattr(cli, "_build_role_backends", explode)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(key, raising=False)
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

def test_resume_candidate_dir_uses_recorded_path_when_present(tmp_path):
    """Прогон, запущенный с --candidate вне CANDIDATES_ROOT, должен грузить
    промпты из ЭТОГО каталога на resume, а не из candidates/<candidate_id>,
    где может лежать другой кандидат с тем же именем."""
    # Arrange
    recorded_dir = tmp_path / "elsewhere" / "my-candidate"
    recorded_dir.mkdir(parents=True)
    state = RunState(run_id="r", candidate_id="my-candidate",
                     config={"candidate_dir": str(recorded_dir)},
                     original_idea="и")

    # Act
    result = cli._resume_candidate_dir(state)

    # Assert
    assert result == recorded_dir


def test_resume_candidate_dir_falls_back_when_key_absent(tmp_path, monkeypatch):
    """state.json старого прогона без candidate_dir — прежнее поведение."""
    monkeypatch.setattr(cli, "CANDIDATES_ROOT", tmp_path / "candidates")
    state = RunState(run_id="r", candidate_id="gen000-generic", config={},
                     original_idea="и")

    result = cli._resume_candidate_dir(state)

    assert result == tmp_path / "candidates" / "gen000-generic"


def test_resume_candidate_dir_fails_clearly_when_recorded_dir_missing(tmp_path):
    state = RunState(run_id="r", candidate_id="my-candidate",
                     config={"candidate_dir": str(tmp_path / "gone")},
                     original_idea="и")

    with pytest.raises(FileNotFoundError) as exc:
        cli._resume_candidate_dir(state)

    assert str(tmp_path / "gone") in str(exc.value)


def test_resume_refuses_finished_run(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setattr(cli, "_build_role_backends", explode)
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


def test_resume_of_old_shape_run_fails_clearly(tmp_path, monkeypatch):
    """state.json прогона, начатого до сменных бэкендов, несёт секцию models:
    продолжать такой прогон нечем — но сообщение должно быть внятным."""
    monkeypatch.setattr(cli, "CANDIDATES_ROOT", Path("candidates"))
    run_dir = tmp_path / "2026-08-03-1215-idea"
    make_state_file(run_dir, config={"loop": {"max_iterations": 6},
                                     "models": {"analyzer": "claude-sonnet-5"}})

    with pytest.raises(SystemExit) as exc:
        cli.main(["resume", str(run_dir)])

    assert "report" in str(exc.value)


# --- команда run -----------------------------------------------------------

def test_run_needs_no_key_for_subscription_backends(tmp_path, monkeypatch):
    """Прогон целиком на подписке не требует ни одной переменной окружения:
    сборка бэкендов поставляемого кандидата обязана пройти с пустым окружением."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from kaidzen.candidate import load_candidate

    backends = cli._build_role_backends(
        load_candidate(Path("candidates") / "gen000-generic"))

    assert set(backends) == {"analyzer", "researcher", "refiner", "judge",
                             "reporter"}


def test_run_fails_at_startup_when_declared_key_is_missing(tmp_path, monkeypatch,
                                                           candidate):
    """Требуем ровно те ключи, которые объявил кандидат, — и падаем на старте,
    назвав переменную и бэкенд, а не на пятой минуте прогона."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)          # чтобы не подхватить .env репозитория
    paid = with_backend(candidate, "deepseek", {"type": "openai_compat",
                                                "api_key_env": "DEEPSEEK_API_KEY"})

    with pytest.raises(SystemExit) as exc:
        cli._build_role_backends(paid)

    message = str(exc.value)
    assert "DEEPSEEK_API_KEY" in message and "deepseek" in message


def test_run_fails_clearly_when_idea_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_build_role_backends", explode)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    missing = tmp_path / "нет-такого.md"

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", str(missing)])

    assert str(missing) in str(exc.value)


def test_run_refuses_to_overwrite_existing_run_dir(tmp_path, monkeypatch):
    """Два прогона одной идеи в ту же секунду не должны затирать state.json
    друг друга: полностью оплаченный прогон нельзя потерять молча."""
    # Arrange
    monkeypatch.setattr(cli, "_build_role_backends", explode)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path / "runs")
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("Идея.", encoding="utf-8")
    candidate_dir = make_candidate(tmp_path / "cand")
    fixed_now = datetime(2026, 8, 3, 12, 15, 0)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(cli, "datetime", FixedDatetime)
    expected_run_dir = cli.make_run_dir(
        runs_root=tmp_path / "runs", idea_path=idea_path,
        now_str=fixed_now.strftime(cli.RUN_DIR_TIMESTAMP_FORMAT))
    make_state_file(expected_run_dir)

    # Act
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", str(idea_path), "--candidate", str(candidate_dir)])

    # Assert
    assert str(expected_run_dir) in str(exc.value)


def test_run_dir_timestamp_includes_seconds():
    """Секунды в имени каталога сужают окно коллизии, но guard выше — это
    настоящая защита (см. test_run_refuses_to_overwrite_existing_run_dir)."""
    assert "%S" in cli.RUN_DIR_TIMESTAMP_FORMAT


# --- вывод хода и завершение прогона --------------------------------------

class FakeSummaryLLM:
    """Отдаёт готовое резюме и фиктивный расход. В сеть не ходит."""

    def __init__(self, summary: str):
        self.usage = ApiUsage(input_tokens=11, output_tokens=22, web_searches=3)
        self._summary = summary
        self.calls = []

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        return cli.SummaryOutput(summary=self._summary)


def with_reporter(candidate, model: str = "reporter-model"):
    """Кандидат с заданной моделью роли reporter (исходный не меняем)."""
    from kaidzen.candidate import RoleConfig
    roles = {**candidate.config.roles,
             cli.REPORTER_ROLE: RoleConfig(backend="subscription", model=model)}
    config = candidate.config.model_copy(update={"roles": roles})
    return candidate.model_copy(update={"config": config})


def with_backend(candidate, name: str, spec: dict, role: str = "judge"):
    """Кандидат, у которого одна роль переехала на дополнительный бэкенд."""
    from kaidzen.candidate import RoleConfig
    backends = {**candidate.config.backends, name: spec}
    roles = {**candidate.config.roles,
             role: RoleConfig(backend=name,
                              model=candidate.config.roles[role].model)}
    config = candidate.config.model_copy(update={"backends": backends,
                                                 "roles": roles})
    return candidate.model_copy(update={"config": config})


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


def test_progress_printer_prints_warning_step(capsys):
    """orchestrator шлёт предупреждения через on_step с префиксом STEP_WARNING —
    без ветки в ProgressPrinter они молча теряются (см. orchestrator._notify)."""
    state = RunState(run_id="r", candidate_id="c", config={}, original_idea="и")
    step = f"{cli.STEP_WARNING}: Researcher вернул находки по неизвестным допущениям: A9"

    cli.ProgressPrinter()(step, state)

    out = capsys.readouterr().out
    assert "Researcher вернул находки по неизвестным допущениям: A9" in out


def test_progress_printer_judge_reports_rollback(capsys):
    state = RunState(run_id="r", candidate_id="c", config={}, original_idea="и",
                     iteration=1,
                     versions=[Version(n=1, idea_text="v1", rolled_back=True)])

    cli.ProgressPrinter()(cli.STEP_JUDGE, state)

    assert "откачена" in capsys.readouterr().out


def make_finished_state(run_dir: Path) -> RunState:
    return RunState(run_id=run_dir.name, candidate_id="gen000-generic",
                    config={}, original_idea="Сырая идея.",
                    stop_reason="plateau", iteration=1,
                    versions=[Version(n=1, idea_text="Доведённая идея.",
                                      judge=make_judge(8.0, 1.5))])


def test_finish_run_writes_report_with_summary_and_prints_usage(
        tmp_path, capsys, candidate):
    # Arrange
    run_dir = tmp_path / "2026-08-03-1215-idea"
    run_dir.mkdir()
    state = make_finished_state(run_dir)
    llm = FakeSummaryLLM("Короткое резюме идеи.")

    # Act
    cli._finish_run(as_backends(llm), with_reporter(candidate), state, run_dir)

    # Assert
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Короткое резюме идеи." in report
    assert "Доведённая идея." in report
    out = capsys.readouterr().out
    assert "Остановка: plateau" in out
    assert "11 входных, 22 выходных" in out and "веб-поисков: 3" in out
    # расход на резюме должен осесть и в state.json, а не только в отчёте
    assert load_state(run_dir).api_usage.input_tokens == 11


class FailingSummaryLLM:
    """Падает на каждом вызове structured — как реальный API-сбой без ретрая."""

    def __init__(self):
        self.usage = ApiUsage(input_tokens=5, output_tokens=5, web_searches=0)

    def structured(self, **kwargs):
        raise RuntimeError("сеть недоступна")


def test_finish_run_writes_report_when_summary_call_fails(tmp_path, capsys, candidate):
    """Единственный незащищённый ретраем вызов не должен хоронить оплаченный
    прогон: отчёт обязан появиться даже если резюме не сгенерировалось."""
    # Arrange
    run_dir = tmp_path / "2026-08-03-1215-idea"
    run_dir.mkdir()
    state = make_finished_state(run_dir)
    llm = FailingSummaryLLM()

    # Act
    cli._finish_run(as_backends(llm), with_reporter(candidate), state, run_dir)

    # Assert
    report_path = run_dir / "report.md"
    assert report_path.exists()
    assert cli.REBUILT_SUMMARY in report_path.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "резюме" in out.lower()
    assert "report.md" in out or str(report_path) in out


def test_summary_model_comes_from_candidate_config(tmp_path, candidate):
    """Имена моделей живут в конфиге кандидата, а не в коде CLI."""
    # Arrange
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    llm = FakeSummaryLLM("резюме")

    # Act
    cli._finish_run(as_backends(llm), with_reporter(candidate, "модель-репортёра"),
                    make_finished_state(run_dir), run_dir)

    # Assert
    assert llm.calls[0]["model"] == "модель-репортёра"


def test_cli_has_no_hardcoded_model_name():
    assert not hasattr(cli, "SUMMARY_MODEL")


def test_summary_call_sends_effort_and_no_temperature(tmp_path, candidate):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    llm = FakeSummaryLLM("резюме")

    cli._finish_run(as_backends(llm), with_reporter(candidate), make_finished_state(run_dir),
                    run_dir)

    call = llm.calls[0]
    assert call["effort"] == cli.SUMMARY_EFFORT
    assert "temperature" not in call


@pytest.mark.parametrize("domain", ["generic", "business", "games"])
def test_shipped_candidates_run_everything_on_subscription(domain):
    """Из коробки проект работает без единого ключа — все роли на подписке."""
    from kaidzen.candidate import load_candidate
    candidate = load_candidate(Path("candidates") / f"gen000-{domain}")
    assert candidate.config.roles[cli.REPORTER_ROLE].model.strip()
    assert {cfg.backend for cfg in candidate.config.roles.values()} == {
        "subscription"}


def test_print_start_shows_run_and_candidate(candidate, capsys):
    cli._print_start(Path("runs/2026-08-03-1215-idea"), candidate)

    out = capsys.readouterr().out
    assert "2026-08-03-1215-idea" in out
    assert candidate.candidate_id in out


def test_print_start_names_backend_of_every_role(candidate, capsys):
    """Пользователь должен видеть, что цикл идёт по подписке, а не по ключу."""
    cli._print_start(Path("runs/2026-08-03-1215-idea"), candidate)

    out = capsys.readouterr().out
    for role, cfg in candidate.config.roles.items():
        assert f"{role}: {cfg.backend} / {cfg.model}" in out


def test_print_start_accepts_max_iterations_override(candidate, capsys):
    """При resume лимит берётся из снапшота state.config, а не из кандидата —
    иначе прогон, начатый с --max-iter, печатал бы чужое число на resume."""
    cli._print_start(Path("runs/2026-08-03-1215-idea"), candidate,
                     max_iterations=99)

    out = capsys.readouterr().out
    assert "до 99 итераций" in out
    assert f"до {candidate.config.loop.max_iterations} итераций" not in out


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
