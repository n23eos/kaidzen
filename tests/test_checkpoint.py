"""Чекпоинты человека: holdout, сводка, approve/reject и блокировка поколений."""
import pytest

from kaidzen.candidate import CHAMPION_PREFIX
from kaidzen.checkpoint import (OK_NOTE, OVERFIT_NOTE, STATUS_APPROVED,
                                STATUS_PENDING, STATUS_REJECTED, ChampionRef,
                                HoldoutResult, approve, make_checkpoint,
                                pending_checkpoint)
from kaidzen.evolve import (STOP_CHECKPOINT_PENDING, STOP_CHECKPOINT_REJECTED,
                            approve_checkpoint, load_evolve_state,
                            reject_checkpoint, run_evolve)
from kaidzen.metrics import RunMetrics
from tests.evolve_fakes import DOMAIN, FakeMeta, FakePipeline, make_ctx, make_env

CHAMPION_ID = "gen000-test"


def evolve_until_checkpoint(env, *, holdout_closed=None, generations=3):
    """Одно поколение с промоцией, после которого открывается чекпоинт."""
    ctx = make_ctx(env, FakeMeta(comparison="challenger"),
                   FakePipeline(closed={CHAMPION_ID: 0.5}, default=0.75,
                                holdout_closed=holdout_closed))
    state = run_evolve(ctx, champion_dir=env.champion_dir,
                       max_generations=generations, checkpoint_every=1)
    return ctx, state


def test_checkpoint_pauses_evolve_and_records_the_pending_decision(tmp_path):
    env = make_env(tmp_path)
    _, state = evolve_until_checkpoint(env)

    assert state.stop_reason == STOP_CHECKPOINT_PENDING
    record = pending_checkpoint(state)
    assert record is not None and record.status == STATUS_PENDING
    assert record.generation == 1


def test_checkpoint_summary_contains_both_reports_and_metrics(tmp_path):
    env = make_env(tmp_path)
    _, state = evolve_until_checkpoint(env)
    text = (env.evolve_dir / "checkpoints" / "gen001.md").read_text(
        encoding="utf-8")

    assert f"Отчёты holdout: {CHAMPION_ID}" in text
    assert f"Отчёты holdout: {state.champion_id}" in text
    assert text.count("Доведённая идея") >= 2 or text.count("доведённая идея") >= 2
    assert "assumptions_closed_rate" in text
    assert "Метрики на holdout" in text and "Метрики на train" in text


def test_holdout_regression_is_flagged_in_summary(tmp_path):
    """Лучше на train, хуже на holdout — подгонка под бенчмарк, и это видно."""
    env = make_env(tmp_path)
    _, state = evolve_until_checkpoint(
        env, holdout_closed={CHAMPION_ID: 0.5, "gen001-a": 0.25,
                             "gen001-b": 0.25})
    text = (env.evolve_dir / "checkpoints" / "gen001.md").read_text(
        encoding="utf-8")

    assert OVERFIT_NOTE in text
    assert state.checkpoints[-1].overfit is True


def test_no_flag_when_holdout_confirms_the_gain(tmp_path):
    env = make_env(tmp_path)
    _, state = evolve_until_checkpoint(
        env, holdout_closed={CHAMPION_ID: 0.5, "gen001-a": 1.0,
                             "gen001-b": 1.0})
    text = (env.evolve_dir / "checkpoints" / "gen001.md").read_text(
        encoding="utf-8")

    assert OK_NOTE in text
    assert state.checkpoints[-1].overfit is False


def test_evolve_refuses_to_continue_while_checkpoint_pending(tmp_path):
    env = make_env(tmp_path)
    ctx, _ = evolve_until_checkpoint(env)
    with pytest.raises(ValueError, match="ждёт решения человека"):
        run_evolve(ctx, resume=True)


def test_approve_advances_generation(tmp_path):
    env = make_env(tmp_path)
    ctx, _ = evolve_until_checkpoint(env)
    record = approve_checkpoint(env.evolve_dir)
    assert record.status == STATUS_APPROVED

    state = run_evolve(ctx, resume=True)
    assert state.generation == 2
    assert len(state.checkpoints) == 2


def test_reject_rolls_champion_back_to_previous(tmp_path):
    env = make_env(tmp_path)
    ctx, promoted = evolve_until_checkpoint(env)
    assert promoted.champion_id != CHAMPION_ID

    record = reject_checkpoint(env.evolve_dir)
    assert record.status == STATUS_REJECTED
    state = load_evolve_state(env.evolve_dir)
    assert state.champion_id == CHAMPION_ID
    pointer = env.candidates_root / f"{CHAMPION_PREFIX}{DOMAIN}"
    assert pointer.read_text(encoding="utf-8").strip() == CHAMPION_ID


def test_reject_stops_the_evolve_run(tmp_path):
    """Отклонение человека — критерий остановки мета-лупа (ТЗ §4.4)."""
    env = make_env(tmp_path)
    ctx, _ = evolve_until_checkpoint(env)
    reject_checkpoint(env.evolve_dir)
    assert run_evolve(ctx, resume=True).stop_reason == STOP_CHECKPOINT_REJECTED


def test_holdout_runs_do_not_overwrite_train_metrics(tmp_path):
    """Иначе чекпоинту нечего сравнивать: обе колонки станут одинаковыми."""
    env = make_env(tmp_path)
    _, state = evolve_until_checkpoint(
        env, holdout_closed={CHAMPION_ID: 0.25, "gen001-a": 0.25,
                             "gen001-b": 0.25})
    assert state.train_metrics[CHAMPION_ID].assumptions_closed_rate == 0.5


def test_first_checkpoint_without_previous_champion_still_writes_summary(tmp_path):
    """Промоции не было — сравнивать не с кем, но пауза всё равно нужна."""
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="champion"), FakePipeline())
    state = run_evolve(ctx, champion_dir=env.champion_dir, max_generations=3,
                       checkpoint_every=1)

    text = (env.evolve_dir / "checkpoints" / "gen001.md").read_text(
        encoding="utf-8")
    assert "Предыдущий чемпион: —" in text
    assert state.checkpoints[-1].overfit is False


def test_approve_without_pending_checkpoint_is_an_error(tmp_path):
    env = make_env(tmp_path)
    ctx = make_ctx(env, FakeMeta(comparison="champion"), FakePipeline())
    state = run_evolve(ctx, champion_dir=env.champion_dir, max_generations=1,
                       checkpoint_every=0)
    with pytest.raises(ValueError, match="нет незакрытого чекпоинта"):
        approve(state)


def test_summary_handles_missing_metrics(tmp_path):
    """Чекпоинт без прогонов не должен падать на форматировании."""
    record = make_checkpoint(
        evolve_dir=tmp_path, generation=7,
        champion=ChampionRef(candidate_id="gen007-a", candidate_dir="x"),
        previous=None, evaluate=lambda ref: HoldoutResult(),
        train_metrics={"gen007-a": RunMetrics()})
    text = (tmp_path / "checkpoints" / "gen007.md").read_text(encoding="utf-8")
    assert record.overfit is False
    assert "—" in text
