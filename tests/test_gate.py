from kaidzen.gate import MIN_WIN_RATE, GateDecision, decide
from kaidzen.metrics import RunMetrics


def m(closed=0.6, partial=0.2, sources=10, facts=10, grounded=1.0, limit=False):
    return RunMetrics(runs=5, high_total=10, high_closed=int(closed * 10),
                      assumptions_closed_rate=closed, partial_rate=partial,
                      facts_total=facts, facts_with_sources=sources,
                      grounded_changelog_rate=grounded, hit_iteration_limit=limit)


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
    d = decide(champion=m(closed=0.60), challenger=m(closed=0.57), win_rate=0.8)
    assert d.promote is True


def test_reason_is_filled_on_promotion_too():
    d = decide(champion=m(), challenger=m(closed=0.8), win_rate=0.8)
    assert d.reason


def test_decide_does_not_mutate_inputs():
    champion, challenger = m(), m(closed=0.9)
    before = (champion.model_dump(), challenger.model_dump())
    decide(champion=champion, challenger=challenger, win_rate=0.9)
    assert (champion.model_dump(), challenger.model_dump()) == before
