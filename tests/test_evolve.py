"""Оркестратор поколений: промоция, слепота, остановки и возобновление."""
import pytest

from kaidzen.evolve import (STOP_MAX_GENERATIONS, STOP_PLATEAU, STOP_REQUESTED,
                            STAGE_GATED, STATUS_PROMOTED, STATUS_UNSTABLE,
                            EVAL_MAX_ITERATIONS, load_evolve_state, request_stop,
                            run_evolve)
from kaidzen.candidate import CHAMPION_PREFIX
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
    assert broken.candidate_id == "gen001-a"
    assert broken.status == STATUS_UNSTABLE
    assert sum(1 for r in broken.runs if not r.ok) >= 2
    assert broken.win_rate is None          # до попарок дело не дошло
    assert state.champion_id == "gen001-b"  # второй челленджер прошёл


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
    assert not (env.candidates_root / "gen001-a").exists()


def test_invalid_mutation_is_discarded_and_leaves_nothing_on_disk(tmp_path):
    from kaidzen.roles.meta.mutator import MutationProposal

    env = make_env(tmp_path)
    broken = MutationProposal(config={"loop": {"max_iterations": 999}},
                              rationale="сломанный конфиг")
    ctx = make_ctx(env, FakeMeta(proposal=broken), FakePipeline())
    state = start(ctx, env)

    assert state.generations[0].challengers == []
    assert not (env.candidates_root / "gen001-a").exists()
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
    assert [c.candidate_id for c in interrupted.challengers] == ["gen001-a"]

    state = run_evolve(make_ctx(env, FakeMeta(comparison="champion"),
                                FakePipeline()), resume=True)
    gen = state.generations[0]
    assert [c.candidate_id for c in gen.challengers] == ["gen001-a", "gen001-b"]
    assert gen.discarded == []
