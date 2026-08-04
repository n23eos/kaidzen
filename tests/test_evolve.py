"""Оркестратор поколений: промоция, слепота, остановки и возобновление."""
from types import SimpleNamespace

import pytest

from kaidzen import evolve, evolve_memory

from kaidzen.evolve import (STOP_MAX_GENERATIONS, STOP_PLATEAU, STOP_REQUESTED,
                            STAGE_GATED, STATUS_PROMOTED, STATUS_UNSTABLE,
                            EVAL_MAX_ITERATIONS, load_evolve_state, request_stop,
                            run_evolve)
from kaidzen.candidate import CHAMPION_PREFIX
from kaidzen.metrics import RunMetrics
from kaidzen.evolution_log import (OUTCOME_DISCARDED, OUTCOME_PROMOTED,
                                   OUTCOME_REJECTED, OUTCOME_UNSTABLE,
                                   load_records)
from kaidzen.roles.meta.diagnostician import Diagnosis
from kaidzen.roles.meta.mutator import MutationProposal
from tests.evolve_fakes import (DOMAIN, FakeMeta, FakePipeline, make_ctx,
                                make_env)

CHAMPION_ID = "gen000-test"
TRAIN_IDEAS = 3


def start(ctx, env, *, generations=1, checkpoint_every=0):
    """Новый evolve-прогон. checkpoint_every=0 отключает чекпоинты."""
    return run_evolve(ctx, champion_dir=env.champion_dir,
                      max_generations=generations,
                      checkpoint_every=checkpoint_every)


def test_generation_promotes_winner(tmp_path):
    """Челленджер выиграл попарки и не просадил метрики — стал чемпионом."""
    env = make_env(tmp_path)
    pipeline = FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"), pipeline)
    state = start(ctx, env)

    assert state.champion_id != CHAMPION_ID
    assert state.previous_champion_id == CHAMPION_ID
    promoted = [c for c in state.generations[0].challengers
                if c.status == STATUS_PROMOTED]
    assert len(promoted) == 1
    assert state.generations[0].promoted_id == promoted[0].candidate_id


def test_promotion_moves_the_champion_pointer(tmp_path):
    """Результат эволюции доходит до обычной работы только через указатель."""
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75))
    state = start(ctx, env)

    pointer = env.candidates_root / f"{CHAMPION_PREFIX}{DOMAIN}"
    assert pointer.read_text(encoding="utf-8").strip() == state.champion_id


def test_generation_keeps_champion_when_gate_rejects(tmp_path):
    """Выиграл попарки, но упал assumptions_closed_rate — чемпион не меняется."""
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(closed={CHAMPION_ID: 1.0}, default=0.25))
    state = start(ctx, env)

    assert state.champion_id == CHAMPION_ID
    reasons = [c.gate_reason for c in state.generations[0].challengers]
    assert all("assumptions_closed_rate" in r for r in reasons)
    assert all(c.win_rate == 1.0 for c in state.generations[0].challengers)


def test_swapped_agreement_counts_as_win(tmp_path):
    """Судья выбрал челленджера в обеих позициях — это победа."""
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"), FakePipeline())
    state = start(ctx, env)
    assert all(c.win_rate == 1.0 for c in state.generations[0].challengers)


def test_pairwise_disagreement_counts_as_tie(tmp_path):
    """A/B дал одного победителя, B/A — другого: ничья, не победа."""
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="disagree"), FakePipeline())
    state = start(ctx, env)

    assert all(c.win_rate == 0.0 for c in state.generations[0].challengers)
    assert state.champion_id == CHAMPION_ID


def test_every_pair_is_judged_twice(tmp_path):
    """Перестановка позиций обязательна: иначе побеждает предвзятость судьи."""
    env = make_env(tmp_path)
    meta = FakeMeta(comparison="champion")
    start(make_ctx(env, meta, FakePipeline()), env)
    challengers = 2
    assert len(meta.comparison_payloads()) == TRAIN_IDEAS * challengers * 2


def test_meta_judge_never_receives_candidate_ids(tmp_path):
    """Сквозная проверка слепоты на уровне оркестратора, а не только роли."""
    env = make_env(tmp_path)
    meta = FakeMeta(comparison="challenger")
    state = start(make_ctx(env, meta, FakePipeline()), env)

    known_ids = [CHAMPION_ID] + [c.candidate_id
                                 for c in state.generations[0].challengers]
    for payload in meta.comparison_payloads():
        for candidate_id in known_ids:
            assert candidate_id not in payload
        assert "gen00" not in payload


def test_two_generations_without_promotion_stop_as_plateau(tmp_path):
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="champion"), FakePipeline())
    state = start(ctx, env, generations=5)

    assert state.stop_reason == STOP_PLATEAU
    assert state.generation == 2
    assert state.champion_id == CHAMPION_ID


def test_max_generations_stops(tmp_path):
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(default=0.75))
    state = start(ctx, env, generations=2)

    assert state.stop_reason == STOP_MAX_GENERATIONS
    assert state.generation == 2


def test_evolve_state_is_saved_after_every_stage(tmp_path):
    """Прерывание в любой точке оставляет валидный state для evolve-resume."""
    env = make_env(tmp_path)
    seen = []

    def observe():
        if (env.evolve_dir / "state.json").exists():
            saved = load_evolve_state(env.evolve_dir)
            seen.append(saved.generations[-1].stage if saved.generations else "")

    ctx = make_ctx(env, FakeMeta(comparison="challenger", observer=observe),
                   FakePipeline())
    state = start(ctx, env)

    assert {"started", "diagnosed", "evaluated"} <= set(seen)
    assert state.generations[-1].stage == STAGE_GATED
    assert load_evolve_state(env.evolve_dir).stop_reason == STOP_MAX_GENERATIONS


def test_resume_skips_completed_evaluations(tmp_path):
    """Уже прогнанные идеи не гоняются повторно — они дорогие по времени."""
    env = make_env(tmp_path)
    first = FakePipeline(interrupt_after=2)
    with pytest.raises(KeyboardInterrupt):
        start(make_ctx(env, FakeMeta(), first), env)
    done = set(first.calls[:2])

    second = FakePipeline()
    state = run_evolve(make_ctx(env, FakeMeta(comparison="challenger"), second),
                       resume=True)

    assert not (set(second.calls) & done)
    assert state.generations[-1].stage == STAGE_GATED


def test_champion_runs_are_reused_from_cache(tmp_path):
    """Чемпион уже прогонялся на этих идеях — берём результат, не гоняем снова."""
    env = make_env(tmp_path)
    pipeline = FakePipeline()
    ctx = make_ctx(env, FakeMeta(comparison="champion"), pipeline)
    state = start(ctx, env, generations=5)

    assert state.generation == 2         # два поколения, чемпион тот же
    champion_calls = [c for c in pipeline.calls if c[0] == CHAMPION_ID]
    assert len(champion_calls) == TRAIN_IDEAS


def test_stop_flag_finishes_current_generation_then_exits(tmp_path):
    """evolve-stop не бросает оплаченные прогоны на полпути."""
    env = make_env(tmp_path)

    class StoppingPipeline(FakePipeline):
        def __call__(self, *args, **kwargs):
            request_stop(env.evolve_dir)
            return super().__call__(*args, **kwargs)

    ctx = make_ctx(env, FakeMeta(comparison="challenger"), StoppingPipeline())
    state = start(ctx, env, generations=5)

    assert state.stop_reason == STOP_REQUESTED
    assert state.generation == 1
    assert state.generations[0].stage == STAGE_GATED
    assert state.stop_requested is True


def test_failed_run_marks_idea_and_two_failures_reject_candidate(tmp_path):
    """Нестабильный кандидат отклоняется, а не тащится дальше."""
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(fails={"gen001-a"}))
    state = start(ctx, env)

    broken = state.generations[0].challengers[0]
    assert broken.candidate_id.startswith("gen001-a")
    assert broken.status == STATUS_UNSTABLE
    assert sum(1 for r in broken.runs if not r.ok) >= 2
    assert broken.win_rate is None          # до попарок дело не дошло
    assert state.champion_id.startswith("gen001-b")  # второй челленджер прошёл


def test_unstable_champion_stops_with_a_clear_error(tmp_path):
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(), FakePipeline(fails={CHAMPION_ID}))
    with pytest.raises(ValueError, match="ни одного успешного прогона"):
        start(ctx, env)


def test_mutation_touching_three_roles_is_discarded(tmp_path):
    """Иначе непонятно, какая из правок дала выигрыш поколения."""
    from kaidzen.roles.meta.mutator import MutationProposal

    env = make_env(tmp_path)
    wide = MutationProposal(prompts={"analyzer": "a", "researcher": "b",
                                     "refiner": "c"}, rationale="широко")
    ctx = make_ctx(env, FakeMeta(comparison="challenger", proposal=wide),
                   FakePipeline())
    state = start(ctx, env)

    gen = state.generations[0]
    assert gen.challengers == []
    assert len(gen.discarded) == 2
    assert "не больше 2" in gen.discarded[0]
    assert not list(env.candidates_root.glob("gen001-a*"))


def test_invalid_mutation_is_discarded_and_leaves_nothing_on_disk(tmp_path):
    from kaidzen.roles.meta.mutator import MutationProposal

    env = make_env(tmp_path)
    broken = MutationProposal(config={"loop": {"max_iterations": 999}},
                              rationale="сломанный конфиг")
    ctx = make_ctx(env, FakeMeta(proposal=broken), FakePipeline())
    state = start(ctx, env)

    assert state.generations[0].challengers == []
    assert not list(env.candidates_root.glob("gen001-a*"))
    assert state.champion_id == CHAMPION_ID


def test_eval_runs_use_the_reduced_iteration_limit(tmp_path):
    """Поколение из десятка прогонов иначе идёт часами."""
    env = make_env(tmp_path)
    limits = []

    class RecordingPipeline(FakePipeline):
        def __call__(self, candidate, **kwargs):
            limits.append(candidate.config.loop.max_iterations)
            return super().__call__(candidate, **kwargs)

    start(make_ctx(env, FakeMeta(), RecordingPipeline()), env)
    assert limits and all(limit <= EVAL_MAX_ITERATIONS for limit in limits)


def test_parallel_eval_gives_the_same_state_as_sequential(tmp_path, tmp_path_factory):
    """Пул не должен терять прогоны и путать состояние."""
    results = []
    for concurrency in (1, 2):
        env = make_env(tmp_path_factory.mktemp(f"c{concurrency}"))
        ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                       FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75),
                       concurrency=concurrency)
        state = start(ctx, env)
        results.append((state.champion_id,
                        sorted(r.idea.split("/")[-1]
                               for c in state.generations[0].challengers
                               for r in c.runs)))
    assert results[0] == results[1]


def test_evolve_run_is_resumable_from_disk_only(tmp_path):
    """Никаких скрытых состояний в памяти: resume читает только state.json."""
    env = make_env(tmp_path)
    start(make_ctx(env, FakeMeta(comparison="champion"), FakePipeline()), env)
    saved = load_evolve_state(env.evolve_dir)
    assert saved.generations[0].champion is not None
    assert saved.cache and all(run.run_dir for run in saved.cache.values())


def test_new_run_requires_a_champion_dir(tmp_path):
    env = make_env(tmp_path)
    with pytest.raises(ValueError, match="champion_dir"):
        run_evolve(make_ctx(env, FakeMeta(), FakePipeline()))


def test_missing_state_is_a_clear_error(tmp_path):
    env = make_env(tmp_path)
    with pytest.raises(FileNotFoundError, match="state.json"):
        run_evolve(make_ctx(env, FakeMeta(), FakePipeline()), resume=True)


def test_summary_shows_the_evolution_line(tmp_path):
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75))
    state = start(ctx, env)
    text = (env.evolve_dir / "summary.md").read_text(encoding="utf-8")

    assert state.champion_id in text
    assert "Поколение 1" in text
    assert "по гипотезе 1" in text       # замысел мутации виден человеку


def test_progress_events_do_not_break_the_generation(tmp_path):
    """Упавший колбэк прогресса не должен ронять оплаченное поколение."""
    import dataclasses

    env = make_env(tmp_path)
    ctx = dataclasses.replace(
        make_ctx(env, FakeMeta(comparison="challenger"), FakePipeline()),
        on_event=lambda message: 1 / 0)
    assert start(ctx, env).generations[0].stage == STAGE_GATED


def test_resume_does_not_repeat_a_finished_mutation(tmp_path):
    """Прерывание между двумя мутациями не должно порождать дубль первой."""
    env = make_env(tmp_path)
    # вызовы мета-уровня: 1 — диагност, 2 — мутация A, 3 — мутация B
    with pytest.raises(KeyboardInterrupt):
        start(make_ctx(env, FakeMeta(interrupt_after=2), FakePipeline()), env)
    interrupted = load_evolve_state(env.evolve_dir).generations[0]
    assert [c.candidate_id.split("-")[1] for c in interrupted.challengers] == ["a"]

    state = run_evolve(make_ctx(env, FakeMeta(comparison="champion"),
                                FakePipeline()), resume=True)
    gen = state.generations[0]
    assert [c.candidate_id.split("-")[1] for c in gen.challengers] == ["a", "b"]
    assert gen.discarded == []


# --- коллизия имён между evolve-прогонами (поймано вживую) ------------------

def test_candidate_id_is_scoped_to_the_evolve_run():
    """Второй прогон не должен упираться в кандидатов, созданных первым."""
    first = evolve._candidate_id(1, 0, "2026-08-03-045447-business")
    second = evolve._candidate_id(1, 0, "2026-08-03-140831-business")
    assert first != second
    assert first.startswith("gen001-a")


def test_candidate_id_is_stable_within_one_run():
    """Resume обязан попасть в то же имя, иначе потомок задвоится."""
    once = evolve._candidate_id(2, 1, "2026-08-03-140831-business")
    twice = evolve._candidate_id(2, 1, "2026-08-03-140831-business")
    assert once == twice
    assert once.startswith("gen002-b")


def test_candidate_id_survives_evolve_id_without_digits():
    assert evolve._candidate_id(1, 0, "прогон") .startswith("gen001-a")


# --- база сравнения не должна усыхать (поймано на полном бенчмарке) ---------

def test_champion_is_evaluated_on_every_idea_despite_failures(tmp_path):
    """Чемпион — база сравнения, а не кандидат на оценку.

    Обрыв его прогонов по FAILURE_LIMIT сузил пересечение с челленджерами до
    одной идеи: бенчмарк из пяти выродился в выборку размером с прошлый прогон.
    """
    env = make_env(tmp_path)
    pipeline = FakePipeline(fails={CHAMPION_ID})
    ctx = make_ctx(env, FakeMeta(comparison="champion"), pipeline)
    with pytest.raises(ValueError, match="эволюционировать"):
        run_evolve(ctx, champion_dir=env.champion_dir, max_generations=1)
    champion_ideas = {idea for cid, idea in pipeline.calls if cid == CHAMPION_ID}
    assert len(champion_ideas) == len(env.benchmark.train), (
        "чемпион обязан пройти все train-идеи, иначе сравнивать не с чем")


def test_challenger_still_stops_after_two_failures(tmp_path):
    """Для челленджера отбраковка по нестабильности остаётся."""
    env = make_env(tmp_path)
    pipeline = FakePipeline(fails={"gen001-a"})
    ctx = make_ctx(env, FakeMeta(), pipeline)
    state = run_evolve(ctx, champion_dir=env.champion_dir, max_generations=1)
    broken = next(c for c in state.generations[-1].challengers
                  if c.candidate_id.startswith("gen001-a"))
    assert broken.status == STATUS_UNSTABLE


def test_compare_reports_how_many_ideas_were_comparable(tmp_path):
    """«побед 0 из 1» при бенчмарке из пяти идей выглядит нормально и молчит
    о том, что выборка схлопнулась. Доля сравнимых идей должна быть в строке."""
    env = make_env(tmp_path)
    events: list[str] = []
    ctx = make_ctx(env, FakeMeta(), FakePipeline(), on_event=events.append)
    run_evolve(ctx, champion_dir=env.champion_dir, max_generations=1)
    lines = [e for e in events if "побед" in e]
    assert lines
    total = len(env.benchmark.train)
    for line in lines:
        assert f"сравнимых идей {total} из {total}" in line, line


def test_compare_warns_when_comparison_set_is_too_thin(tmp_path):
    """Мало сравнимых идей — предупреждение, а не молчаливый результат."""
    env = make_env(tmp_path)
    # чемпион падает на всех идеях кроме одной: сравнивать будет нечего
    pipeline = FakePipeline(fails=set())
    events: list[str] = []
    ctx = make_ctx(env, FakeMeta(), pipeline, on_event=events.append)
    run_evolve(ctx, champion_dir=env.champion_dir, max_generations=1)
    assert evolve.MIN_COMPARABLE_IDEAS >= 2


def test_eval_run_writes_a_readable_report(tmp_path):
    """Рядом с состоянием должен лежать читаемый отчёт.

    Без него результат поколения приходится доставать из state.json руками —
    а собрать отчёт можно бесплатно, модель для этого не нужна.
    """
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(), FakePipeline())
    run_evolve(ctx, champion_dir=env.champion_dir, max_generations=1)
    reports = list((env.evolve_dir / "runs").rglob("report.md"))
    states = list((env.evolve_dir / "runs").rglob("state.json"))
    assert reports, "ни одного отчёта не собрано"
    assert len(reports) == len(states), "отчёт есть не у каждого прогона"
    text = reports[0].read_text(encoding="utf-8")
    assert "## Допущения" in text and "## Next steps" in text


def test_failed_eval_run_does_not_break_on_missing_report(tmp_path):
    """Упавший прогон отчёта не оставляет — и это не должно ронять поколение."""
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(), FakePipeline(fails={"gen001-a"}))
    state = run_evolve(ctx, champion_dir=env.champion_dir, max_generations=1)
    assert state.generations[-1].challengers


# --- память эволюции и эффективность ---------------------------------------

def log_of(env):
    return load_records(env.candidates_root, DOMAIN)


def test_gate_writes_a_record_per_challenger(tmp_path):
    """Каждая попытка мутации оставляет след, независимо от исхода."""
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75))
    state = start(ctx, env)

    records = log_of(env)
    assert len(records) == len(state.generations[0].challengers)
    outcomes = {r.outcome for r in records}
    assert OUTCOME_PROMOTED in outcomes and OUTCOME_REJECTED in outcomes
    promoted = [r for r in records if r.outcome == OUTCOME_PROMOTED]
    assert [r.candidate_id for r in promoted] == [state.champion_id]


def test_record_carries_hypothesis_roles_and_deltas(tmp_path):
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75))
    state = start(ctx, env)

    record = [r for r in log_of(env) if r.outcome == OUTCOME_PROMOTED][0]
    assert record.hypothesis == "по гипотезе 1"
    assert record.roles_touched == ["researcher"]
    assert record.parent_id == CHAMPION_ID
    assert record.evolve_id == state.evolve_id
    assert record.win_rate == 1.0
    assert record.comparable_ideas == TRAIN_IDEAS
    assert record.metrics_delta["assumptions_closed_rate"] > 0
    assert "output_tokens" in record.metrics_delta
    assert record.gate_reason


def test_discarded_mutation_is_recorded_too(tmp_path):
    """Патч, не доехавший до диска, — тоже знание: не повторять."""
    env = make_env(tmp_path)
    bad = MutationProposal(prompts={"нет-такой-роли": "текст"}, rationale="замысел")
    ctx = make_ctx(env, FakeMeta(comparison="challenger", proposal=bad),
                   FakePipeline())
    start(ctx, env)

    records = log_of(env)
    assert len(records) == 2
    assert all(r.outcome == OUTCOME_DISCARDED for r in records)
    assert all("нет-такой-роли" in r.gate_reason for r in records)
    assert all(r.hypothesis == "замысел" for r in records)


def test_unstable_candidate_is_recorded_as_unstable(tmp_path):
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(fails=["gen001-a"]))
    start(ctx, env)

    outcomes = {r.outcome for r in log_of(env)
                if r.candidate_id.startswith("gen001-a")}
    assert outcomes == {OUTCOME_UNSTABLE}


def test_second_generation_appends_instead_of_overwriting(tmp_path):
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="champion"), FakePipeline())
    start(ctx, env, generations=2)

    records = log_of(env)
    assert len(records) == 4
    assert {r.generation for r in records} == {1, 2}


def test_diagnostician_sees_the_log_of_the_previous_run(tmp_path):
    """Второй evolve-прогон стартует уже не с нуля."""
    env = make_env(tmp_path)
    first = FakeMeta(comparison="champion")
    start(make_ctx(env, first, FakePipeline()), env)

    second_dir = tmp_path / "evolve" / "e2"
    env2 = SimpleNamespace(**{**vars(env), "evolve_dir": second_dir})
    second = FakeMeta(comparison="champion")
    start(make_ctx(env2, second, FakePipeline()), env2)

    diagnosis_calls = [c["user"] for c in second.calls
                       if c["schema"] is Diagnosis]
    assert diagnosis_calls and "Журнал эволюции" in diagnosis_calls[0]
    assert "по гипотезе 1" in diagnosis_calls[0]
    # первый прогон видеть было нечего
    assert all("Журнал эволюции" not in c["user"] for c in first.calls
               if c["schema"] is Diagnosis)


def test_mutator_gets_the_do_not_break_list_of_earlier_findings(tmp_path):
    env = make_env(tmp_path)
    start(make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75)), env)

    second_dir = tmp_path / "evolve" / "e2"
    env2 = SimpleNamespace(**{**vars(env), "evolve_dir": second_dir})
    second = FakeMeta(comparison="champion")
    start(make_ctx(env2, second, FakePipeline()), env2)

    mutator_calls = [c["user"] for c in second.calls
                     if c["schema"] is MutationProposal]
    assert mutator_calls and "не ломать" in mutator_calls[0]


def test_judge_never_sees_the_evolution_log(tmp_path):
    """Слепота судьи держится и при полном журнале за спиной."""
    env = make_env(tmp_path)
    start(make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75)), env)

    second_dir = tmp_path / "evolve" / "e2"
    env2 = SimpleNamespace(**{**vars(env), "evolve_dir": second_dir})
    second = FakeMeta(comparison="challenger")
    start(make_ctx(env2, second, FakePipeline()), env2)

    assert log_of(env2)      # журнал непустой — есть чему утекать
    for payload in second.comparison_payloads():
        for leak in ("Журнал эволюции", "не ломать", "гипотез", "promoted"):
            assert leak not in payload, f"судье утекло: {leak}"


def test_tie_break_promotes_the_cheaper_challenger(tmp_path):
    """Оба прошли Gate с равным win_rate — чемпионом становится дешёвый."""
    env = make_env(tmp_path)
    pipeline = FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75,
                            tokens={"gen001-a": 9_000, "gen001-b": 4_000},
                            default_tokens=8_000)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"), pipeline)
    state = start(ctx, env)

    assert state.champion_id.startswith("gen001-b")
    loser = [c for c in state.generations[0].challengers
             if c.candidate_id != state.champion_id][0]
    assert "дешевле" in loser.gate_reason


def test_expensive_challenger_does_not_become_champion(tmp_path):
    """Улучшение втрое большей ценой Gate не пропускает."""
    env = make_env(tmp_path)
    pipeline = FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.9,
                            tokens={CHAMPION_ID: 5_000}, default_tokens=20_000)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"), pipeline)
    state = start(ctx, env)

    assert state.champion_id == CHAMPION_ID
    assert all("output_tokens" in c.gate_reason
               for c in state.generations[0].challengers)


def test_cheaper_and_better_challenger_still_promotes(tmp_path):
    env = make_env(tmp_path)
    pipeline = FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.9,
                            tokens={CHAMPION_ID: 20_000}, default_tokens=5_000)
    ctx = make_ctx(env, FakeMeta(comparison="challenger"), pipeline)
    state = start(ctx, env)

    assert state.champion_id != CHAMPION_ID
    record = [r for r in log_of(env) if r.outcome == OUTCOME_PROMOTED][0]
    assert record.metrics_delta["output_tokens"] < 0


def test_repeated_gate_after_resume_does_not_duplicate_records(tmp_path):
    """Стадия Gate может повториться при resume — журнал не должен раздваиваться."""
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="champion"), FakePipeline())
    state = start(ctx, env)

    gen = state.generations[0]
    evolve._gate(ctx, state, gen)
    assert len(log_of(env)) == 2


def challenger(candidate_id, win_rate, tokens, runs=2):
    return evolve.CandidateRecord(
        candidate_id=candidate_id, candidate_dir="/нет", win_rate=win_rate,
        metrics=RunMetrics(runs=runs, output_tokens=tokens))


def test_more_wins_beat_a_cheaper_rival():
    """Цена — тай-брейк, а не главный критерий: качество решает первым."""
    cheap = challenger("дешёвый", win_rate=0.6, tokens=1_000)
    strong = challenger("сильный", win_rate=1.0, tokens=50_000)
    assert evolve._better_challenger(cheap, strong) is strong
    assert evolve._better_challenger(strong, cheap) is strong


def test_loser_is_told_why_it_lost():
    strong = challenger("сильный", win_rate=1.0, tokens=50_000)
    weak = challenger("слабый", win_rate=0.6, tokens=1_000)
    assert "попарок" in evolve._why_lost(strong, weak)
    assert "дешевле" in evolve._why_lost(
        strong, challenger("дорогой", win_rate=1.0, tokens=90_000))


def test_delta_without_metrics_is_empty():
    """Кандидат без единого успешного прогона не даёт дельт, а не нулевые."""
    assert evolve_memory.metrics_delta(None, RunMetrics()) == {}
    assert evolve_memory.metrics_delta(RunMetrics(), None) == {}
