from kaidzen.gate import MAX_COST_GROWTH, MIN_WIN_RATE, GateDecision, decide
from kaidzen.metrics import RunMetrics


def m(closed=0.6, partial=0.2, sources=10, facts=10, grounded=1.0, limit=False):
    return RunMetrics(runs=5, high_total=10, high_closed=int(closed * 10),
                      assumptions_closed_rate=closed, partial_rate=partial,
                      facts_total=facts, facts_with_sources=sources,
                      grounded_changelog_rate=grounded, hit_iteration_limit=limit)


def shrunk(high_total, high_closed, **kw):
    """Кандидат с явно заданным размером реестра и числом закрытых."""
    rate = high_closed / high_total if high_total else 0.0
    return m(closed=rate, **kw).model_copy(update={
        "high_total": high_total, "high_closed": high_closed,
        "assumptions_closed_rate": rate})


def test_promotes_on_wins_and_no_regression():
    d = decide(champion=m(), challenger=m(closed=0.7), win_rate=0.7)
    assert isinstance(d, GateDecision)
    assert d.promote is True


def test_promotes_when_challenger_improves_everything():
    d = decide(champion=m(closed=0.5, partial=0.4, sources=6, grounded=0.5),
               challenger=m(closed=0.8, partial=0.1, sources=10, grounded=1.0),
               win_rate=0.9)
    assert d.promote is True


def test_identical_metrics_are_not_a_regression():
    """Копия чемпиона не должна отклоняться из-за шума в делении float."""
    d = decide(champion=m(), challenger=m(), win_rate=0.8)
    assert d.promote is True


def test_rejects_when_win_rate_below_threshold():
    d = decide(champion=m(), challenger=m(closed=0.9), win_rate=0.5)
    assert d.promote is False
    assert "попарк" in d.reason


def test_win_rate_exactly_at_threshold_passes():
    d = decide(champion=m(), challenger=m(), win_rate=MIN_WIN_RATE)
    assert d.promote is True


def test_rejects_pretty_report_with_worse_closed_rate():
    """Главный сценарий Гудхарта: судья доволен, а работа сделана хуже."""
    d = decide(champion=m(closed=0.6), challenger=m(closed=0.4), win_rate=0.9)
    assert d.promote is False
    assert "assumptions_closed_rate" in d.reason


def test_growing_closed_rate_is_not_a_regression():
    """Направление проверки: рост доли закрытых допущений — это хорошо."""
    d = decide(champion=m(closed=0.4), challenger=m(closed=0.9), win_rate=0.9)
    assert d.promote is True


def test_rejects_when_partial_rate_grows():
    d = decide(champion=m(partial=0.2), challenger=m(partial=0.5), win_rate=0.9)
    assert d.promote is False
    assert "partial_rate" in d.reason


def test_falling_partial_rate_is_not_a_regression():
    """Направление проверки: падение хеджирования — цель мета-лупа."""
    d = decide(champion=m(partial=0.5), challenger=m(partial=0.1), win_rate=0.9)
    assert d.promote is True


def test_partial_rate_growing_from_zero_is_rejected():
    d = decide(champion=m(partial=0.0), challenger=m(partial=0.5), win_rate=0.9)
    assert d.promote is False
    assert "partial_rate" in d.reason


def test_tiny_partial_growth_from_zero_is_tolerated():
    d = decide(champion=m(partial=0.0), challenger=m(partial=0.05), win_rate=0.9)
    assert d.promote is True


def test_rejects_when_facts_lose_sources():
    d = decide(champion=m(sources=10, facts=10),
               challenger=m(sources=6, facts=10), win_rate=0.9)
    assert d.promote is False
    assert "source_rate" in d.reason


def test_rejects_when_grounded_changelog_rate_falls():
    d = decide(champion=m(grounded=1.0), challenger=m(grounded=0.5), win_rate=0.9)
    assert d.promote is False
    assert "grounded_changelog_rate" in d.reason


def test_champion_without_facts_gives_no_source_regression():
    """Не с чем сравнивать — не считаем это просадкой."""
    d = decide(champion=m(sources=0, facts=0), challenger=m(sources=0, facts=0),
               win_rate=0.8)
    assert d.promote is True


def test_all_broken_metrics_are_listed_in_reason():
    d = decide(champion=m(), challenger=m(closed=0.3, partial=0.6, grounded=0.2),
               win_rate=0.9)
    assert d.promote is False
    for name in ("assumptions_closed_rate", "partial_rate",
                 "grounded_changelog_rate"):
        assert name in d.reason


def test_small_regression_within_tolerance_is_allowed():
    """Реестр чуть вырос, закрыто столько же — доля просела в пределах допуска."""
    d = decide(champion=shrunk(10, 6), challenger=shrunk(11, 6), win_rate=0.8)
    assert d.promote is True


def test_reason_is_filled_on_promotion_too():
    d = decide(champion=m(), challenger=m(closed=0.8), win_rate=0.8)
    assert d.reason


def test_decide_does_not_mutate_inputs():
    champion, challenger = m(), m(closed=0.9)
    before = (champion.model_dump(), challenger.model_dump())
    decide(champion=champion, challenger=challenger, win_rate=0.9)
    assert (champion.model_dump(), challenger.model_dump()) == before


# --- Гудхарт, пойманный на первом живом поколении ---------------------------

def test_rejects_rate_gained_by_shrinking_the_registry():
    """Живой случай: 4 из 6 → 4 из 4. Доля выросла, работы не прибавилось."""
    d = decide(champion=shrunk(6, 4), challenger=shrunk(4, 4), win_rate=1.0)
    assert d.promote is False
    assert "реестр" in d.reason


def test_allows_smaller_registry_when_more_is_actually_closed():
    """Ужал реестр, но закрыл больше в штуках — это настоящее улучшение."""
    d = decide(champion=shrunk(6, 4), challenger=shrunk(4, 5), win_rate=0.8)
    assert d.promote is True


def test_allows_same_registry_with_more_closed():
    d = decide(champion=shrunk(6, 4), challenger=shrunk(6, 5), win_rate=0.8)
    assert d.promote is True


def test_rejects_when_absolute_closed_drops():
    d = decide(champion=shrunk(6, 5), challenger=shrunk(6, 3), win_rate=0.9)
    assert d.promote is False


def test_small_registry_change_is_not_treated_as_shrinking():
    d = decide(champion=shrunk(10, 6), challenger=shrunk(9, 6), win_rate=0.8)
    assert d.promote is True


# --- эффективность как цель (ТЗ памяти эволюции §4) -------------------------

def costly(tokens, runs=5, **kw):
    """Кандидат с заданным расходом выходных токенов на пачку прогонов."""
    return m(**kw).model_copy(update={"runs": runs, "output_tokens": tokens})


def test_rejects_challenger_that_bought_quality_at_triple_the_cost():
    """Улучшение втрое дороже — это обмен, и решать его должен человек."""
    d = decide(champion=costly(10_000), challenger=costly(30_000, closed=0.9),
               win_rate=0.9)
    assert d.promote is False
    assert "output_tokens" in d.reason


def test_cheaper_and_better_challenger_promotes():
    d = decide(champion=costly(10_000), challenger=costly(6_000, closed=0.9),
               win_rate=0.9)
    assert d.promote is True


def test_modest_growth_under_the_ceiling_is_allowed():
    d = decide(champion=costly(10_000), challenger=costly(14_000), win_rate=0.8)
    assert d.promote is True


def test_cost_ceiling_counts_tokens_per_run_not_per_batch():
    """Челленджер прогнан по большему числу идей — это не удорожание."""
    d = decide(champion=costly(10_000, runs=2), challenger=costly(25_000, runs=5),
               win_rate=0.8)
    assert d.promote is True


def test_champion_without_token_stats_gives_no_cost_regression():
    d = decide(champion=costly(0), challenger=costly(50_000), win_rate=0.8)
    assert d.promote is True


def test_cost_ceiling_exactly_at_the_limit_passes():
    d = decide(champion=costly(10_000),
               challenger=costly(int(10_000 * MAX_COST_GROWTH)), win_rate=0.8)
    assert d.promote is True
