"""Журнал эволюции: дозапись поверх диска и выжимка для мета-ролей."""
import json

import pytest

from kaidzen.evolution_log import (DIGEST_RECENT_LIMIT, OUTCOME_DISCARDED,
                                   OUTCOME_PROMOTED, OUTCOME_REJECTED,
                                   EvolutionRecord, append_record, digest_records,
                                   load_records, log_path, render_digest,
                                   render_do_not_break)

DOMAIN = "business"


def record(candidate_id="gen001-a-165138", outcome=OUTCOME_PROMOTED, **kw):
    data = dict(evolve_id="2026-08-03-165138-business", generation=1,
                candidate_id=candidate_id, parent_id="gen000-business",
                hypothesis="запретить partial без missing_evidence",
                roles_touched=["researcher"], outcome=outcome,
                gate_reason="выиграл попарки (100%), метрики не просели",
                win_rate=1.0,
                metrics_delta={"assumptions_closed_rate": 0.5,
                               "partial_rate": -0.3, "high_closed": 7.0,
                               "output_tokens": -12000.0},
                comparable_ideas=2)
    data.update(kw)
    return EvolutionRecord(**data)


def test_log_lives_next_to_candidates_and_is_named_by_domain(tmp_path):
    assert log_path(tmp_path, DOMAIN) == tmp_path / "EVOLUTION-business.json"


def test_missing_log_reads_as_empty(tmp_path):
    assert load_records(tmp_path, DOMAIN) == []


def test_append_creates_and_reads_back(tmp_path):
    append_record(tmp_path, DOMAIN, record())
    [saved] = load_records(tmp_path, DOMAIN)
    assert saved.candidate_id == "gen001-a-165138"
    assert saved.metrics_delta["output_tokens"] == -12000.0


def test_second_append_keeps_the_first(tmp_path):
    """Дозапись, а не перезапись: журнал переживает обрыв прогона."""
    append_record(tmp_path, DOMAIN, record(candidate_id="c1"))
    append_record(tmp_path, DOMAIN, record(candidate_id="c2",
                                           outcome=OUTCOME_REJECTED))
    assert [r.candidate_id for r in load_records(tmp_path, DOMAIN)] == ["c1", "c2"]


def test_append_takes_the_file_on_disk_as_truth(tmp_path):
    """Запись чужого процесса не теряется: дописываем к прочитанному с диска."""
    append_record(tmp_path, DOMAIN, record(candidate_id="c1"))
    # как будто параллельный прогон дописал свою запись, пока мы работали
    path = log_path(tmp_path, DOMAIN)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.append(record(candidate_id="чужой").model_dump())
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    append_record(tmp_path, DOMAIN, record(candidate_id="c2"))
    assert [r.candidate_id for r in load_records(tmp_path, DOMAIN)] == [
        "c1", "чужой", "c2"]


def test_repeated_append_of_the_same_attempt_does_not_duplicate(tmp_path):
    """Повтор стадии после resume не должен размножать одну попытку."""
    append_record(tmp_path, DOMAIN, record(candidate_id="c1"))
    append_record(tmp_path, DOMAIN, record(candidate_id="c1"))
    assert len(load_records(tmp_path, DOMAIN)) == 1


def test_same_candidate_id_in_another_run_is_a_separate_record(tmp_path):
    append_record(tmp_path, DOMAIN, record(candidate_id="c1"))
    append_record(tmp_path, DOMAIN, record(candidate_id="c1", evolve_id="другой"))
    assert len(load_records(tmp_path, DOMAIN)) == 2


def test_domains_are_independent(tmp_path):
    append_record(tmp_path, DOMAIN, record())
    assert load_records(tmp_path, "games") == []


def test_broken_log_fails_loudly(tmp_path):
    log_path(tmp_path, DOMAIN).write_text("не json", encoding="utf-8")
    with pytest.raises(ValueError, match="журнал эволюции"):
        load_records(tmp_path, DOMAIN)


def test_unknown_outcome_is_rejected():
    with pytest.raises(ValueError):
        record(outcome="почти получилось")


def test_digest_keeps_all_promoted_and_truncates_the_rest():
    """Удачи не выпадают из памяти никогда, неудачи — по мере разрастания."""
    promoted = [record(candidate_id=f"p{i}") for i in range(3)]
    rejected = [record(candidate_id=f"r{i}", outcome=OUTCOME_REJECTED)
                for i in range(DIGEST_RECENT_LIMIT + 5)]
    kept = digest_records(promoted + rejected)

    assert all(r in kept for r in promoted)
    others = [r for r in kept if r.outcome != OUTCOME_PROMOTED]
    assert len(others) == DIGEST_RECENT_LIMIT
    # усечение оставляет свежие, а не первые попавшиеся
    assert others == rejected[-DIGEST_RECENT_LIMIT:]


def test_digest_of_empty_log_is_empty_text():
    assert render_digest([]) == ""
    assert render_do_not_break([]) == ""


def test_rendered_digest_names_outcome_role_and_hypothesis():
    text = render_digest([record(outcome=OUTCOME_REJECTED,
                                 gate_reason="просадил partial_rate")])
    assert "researcher" in text
    assert "missing_evidence" in text
    assert "просадил partial_rate" in text
    assert OUTCOME_REJECTED in text


def test_rendered_digest_shows_the_token_delta():
    """Дешевеет прогон или дорожает — видно прямо в выжимке."""
    assert "-12000" in render_digest([record()])


def test_discarded_attempt_gets_into_the_digest():
    text = render_digest([record(outcome=OUTCOME_DISCARDED,
                                 gate_reason="патч правит три роли")])
    assert "патч правит три роли" in text


def test_do_not_break_lists_only_promoted():
    text = render_do_not_break([record(candidate_id="хорошая"),
                                record(candidate_id="плохая",
                                       outcome=OUTCOME_REJECTED,
                                       hypothesis="ужать реестр допущений")])
    assert "missing_evidence" in text
    assert "ужать реестр" not in text


def test_do_not_break_warns_that_overwriting_is_a_regression():
    text = render_do_not_break([record()])
    assert "регресс" in text
