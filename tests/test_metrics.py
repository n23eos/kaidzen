import pytest

from kaidzen.metrics import RunMetrics, aggregate, run_metrics
from kaidzen.state import Assumption, ChangelogEntry, Fact, RunState, Version


def state(assumptions=(), versions=(), stop_reason="plateau", iteration=2,
          rollbacks=0):
    return RunState(run_id="r", candidate_id="c", config={}, original_idea="i",
                    assumptions=list(assumptions), versions=list(versions),
                    stop_reason=stop_reason, iteration=iteration,
                    rollbacks=rollbacks)


def a(id_, criticality="high", status="unverified", facts=()):
    return Assumption(id=id_, text="t", criticality=criticality, status=status,
                      facts=list(facts))


FACT = Fact(claim="c", source_url="https://example.com/a")
BAD_FACT = Fact(claim="c", source_url="https://example.com/b")


def test_closed_rate_counts_only_high_criticality():
    m = run_metrics(state([
        a("A1", status="confirmed"), a("A2", status="refuted"),
        a("A3", status="unverified"), a("A4", "low", status="unverified")]))
    assert m.high_total == 3
    assert m.high_closed == 2
    assert m.assumptions_closed_rate == pytest.approx(2 / 3)


def test_partial_is_not_closed_and_has_its_own_rate():
    m = run_metrics(state([a("A1", status="partial"), a("A2", status="confirmed")]))
    assert m.high_closed == 1
    assert m.partial_rate == pytest.approx(0.5)


def test_untestable_counts_as_closed():
    """Непроверяемое поиском — честный результат, а не невыполненная работа."""
    m = run_metrics(state([a("A1", status="untestable")]))
    assert m.assumptions_closed_rate == 1.0


def test_facts_with_sources_rate():
    m = run_metrics(state([a("A1", status="confirmed", facts=[FACT, BAD_FACT])]))
    assert m.facts_total == 2
    assert m.facts_with_sources == 2


def test_zero_assumptions_gives_zero_rates_not_crash():
    m = run_metrics(state())
    assert m.assumptions_closed_rate == 0.0
    assert m.partial_rate == 0.0


def test_stop_reason_and_iterations_are_carried():
    m = run_metrics(state(stop_reason="max_iterations", iteration=6))
    assert m.stop_reason == "max_iterations"
    assert m.iterations == 6
    assert m.hit_iteration_limit is True


def test_plateau_stop_is_not_iteration_limit():
    m = run_metrics(state(stop_reason="plateau"))
    assert m.hit_iteration_limit is False


def test_rollbacks_and_usage_are_carried():
    s = state(rollbacks=3)
    s.api_usage.input_tokens = 100
    s.api_usage.output_tokens = 20
    s.api_usage.web_searches = 4
    m = run_metrics(s)
    assert m.rollbacks == 3
    assert (m.input_tokens, m.output_tokens, m.web_searches) == (100, 20, 4)


def test_grounded_changelog_rate():
    v = Version(n=1, idea_text="v", changelog=[
        ChangelogEntry(change="a", reason="r", grounded_in=["A1"]),
        ChangelogEntry(change="b", reason="r", grounded_in=[])])
    m = run_metrics(state(versions=[v]))
    assert m.grounded_changelog_rate == pytest.approx(0.5)


def test_no_changelog_gives_zero_grounded_rate_not_crash():
    m = run_metrics(state(versions=[Version(n=1, idea_text="v")]))
    assert m.grounded_changelog_rate == 0.0


def test_run_metrics_does_not_mutate_state():
    """Метрики — чистая функция: state после расчёта должен быть тем же."""
    s = state([a("A1", status="partial")])
    before = s.model_dump_json()
    run_metrics(s)
    assert s.model_dump_json() == before


def test_aggregate_averages_across_runs():
    one = run_metrics(state([a("A1", status="confirmed")]))
    two = run_metrics(state([a("A1", status="unverified")]))
    agg = aggregate([one, two])
    assert agg.assumptions_closed_rate == pytest.approx(0.5)
    assert agg.runs == 2


def test_aggregate_sums_counters():
    one = run_metrics(state([a("A1", status="confirmed", facts=[FACT])],
                            iteration=2, rollbacks=1))
    two = run_metrics(state([a("A1", status="confirmed", facts=[BAD_FACT])],
                            iteration=3, rollbacks=2))
    agg = aggregate([one, two])
    assert agg.high_total == 2
    assert agg.high_closed == 2
    assert agg.facts_total == 2
    assert agg.facts_with_sources == 2
    assert agg.iterations == 5
    assert agg.rollbacks == 3


def test_aggregate_flags_iteration_limit_if_any_run_hit_it():
    ok = run_metrics(state(stop_reason="plateau"))
    hit = run_metrics(state(stop_reason="max_iterations"))
    agg = aggregate([ok, hit])
    assert agg.hit_iteration_limit is True
    # общая причина остановки у пачки прогонов бессмысленна
    assert agg.stop_reason is None


def test_aggregate_of_empty_list_does_not_crash():
    agg = aggregate([])
    assert agg == RunMetrics(runs=0)
    assert agg.runs == 0
    assert agg.assumptions_closed_rate == 0.0
