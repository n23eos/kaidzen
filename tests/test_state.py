import os
import pytest
from pathlib import Path
from pydantic import ValidationError
from kaidzen.state import (Assumption, Fact, JudgeResult, RunState, Version,
                           save_state, load_state)


def test_assumption_defaults_to_unverified():
    a = Assumption(id="A1", text="рынок существует", criticality="high")
    assert a.status == "unverified"
    assert a.facts == []


def test_assumption_rejects_bad_criticality():
    with pytest.raises(ValidationError):
        Assumption(id="A1", text="x", criticality="huge")


def test_fact_requires_source_url():
    with pytest.raises(ValidationError):
        Fact(claim="x", source_title="t")


@pytest.mark.parametrize("bad_url", ["", "   ", "из общих знаний", "example.com",
                                     "ftp://example.com/x"])
def test_fact_rejects_non_http_source(bad_url):
    with pytest.raises(ValidationError):
        Fact(claim="x", source_url=bad_url)


@pytest.mark.parametrize("url", ["http://example.com/a", "https://example.com/b"])
def test_fact_accepts_http_source_and_keeps_plain_str(url):
    f = Fact(claim="x", source_url=url)
    assert f.source_url == url
    assert isinstance(f.source_url, str)
    assert Fact.model_validate_json(f.model_dump_json()) == f


def test_judge_verdict_enum():
    with pytest.raises(ValidationError):
        JudgeResult(scores={"clarity": 5}, total=5.0,
                    delta_vs_previous=0.0, critique=[], verdict="maybe")


@pytest.mark.parametrize("score", [-0.5, 10.5, 85.0])
def test_judge_rejects_score_outside_0_10(score):
    with pytest.raises(ValidationError):
        JudgeResult(scores={"clarity": score}, total=score,
                    delta_vs_previous=0.0, critique=[], verdict="continue")


def test_judge_rejects_total_not_matching_scores():
    with pytest.raises(ValidationError, match="total"):
        JudgeResult(scores={"a": 5.0, "b": 4.0}, total=42.0,
                    delta_vs_previous=0.0, critique=[], verdict="continue")


def test_judge_accepts_total_within_tolerance():
    jr = JudgeResult(scores={"a": 5.0, "b": 4.0}, total=9.005,
                     delta_vs_previous=0.0, critique=[], verdict="continue")
    assert jr.total == 9.005


def test_judge_allows_any_number_of_axes():
    """Количество осей проверяет валидация кандидата, не JudgeResult."""
    jr = JudgeResult(scores={"a": 1.0}, total=1.0, delta_vs_previous=0.0,
                     critique=[], verdict="continue")
    assert jr.scores == {"a": 1.0}


def test_runstate_roundtrip():
    s = RunState(run_id="r1", candidate_id="gen000-generic",
                 config={}, original_idea="idea")
    s2 = RunState.model_validate_json(s.model_dump_json())
    assert s2.iteration == 0 and s2.versions == []
    assert s2.pending_findings is None


def test_runstate_keeps_pending_findings_across_save(tmp_path: Path):
    # находки Researcher должны переживать перезапуск: их ждёт Refiner
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="i",
                 pending_findings='{"findings": []}')
    save_state(s, tmp_path)
    assert load_state(tmp_path).pending_findings == '{"findings": []}'


def test_current_version_skips_rolled_back():
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="raw",
                 versions=[Version(n=1, idea_text="v1"),
                           Version(n=2, idea_text="v2", rolled_back=True)])
    assert s.current_version().n == 1
    assert s.current_idea_text() == "v1"


def test_current_idea_text_falls_back_to_original():
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="raw")
    assert s.current_version() is None
    assert s.current_idea_text() == "raw"


def test_save_load_roundtrip(tmp_path: Path):
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="i")
    save_state(s, tmp_path)
    assert load_state(tmp_path) == s


def test_save_leaves_no_tmp_file(tmp_path: Path):
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="i")
    save_state(s, tmp_path)
    assert (tmp_path / "state.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_save_creates_missing_dir(tmp_path: Path):
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="i")
    target = tmp_path / "nested" / "run1"
    save_state(s, target)
    assert (target / "state.json").exists()


def test_save_fsyncs_data_and_directory(tmp_path: Path, monkeypatch):
    """Без fsync rename может лечь раньше данных — state.json окажется обрезан."""
    synced: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(fd))
    dir_fds: list[int] = []
    real_open = os.open

    def spy_open(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        if flags & os.O_DIRECTORY:
            dir_fds.append(fd)
        return fd

    monkeypatch.setattr(os, "open", spy_open)
    save_state(RunState(run_id="r1", candidate_id="c", config={},
                        original_idea="i"), tmp_path)
    assert len(synced) >= 2          # файл + каталог
    assert dir_fds and dir_fds[0] in synced


def test_save_removes_temp_file_when_serialization_fails(tmp_path: Path, monkeypatch):
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="i")
    monkeypatch.setattr(type(s), "model_dump_json",
                        lambda self, **kw: (_ for _ in ()).throw(RuntimeError("бум")))
    with pytest.raises(RuntimeError):
        save_state(s, tmp_path)
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "state.json").exists()


def test_save_uses_unique_temp_names(tmp_path: Path, monkeypatch):
    """Фиксированное имя state.json.tmp — гонка между двумя записями."""
    seen: list[str] = []
    import tempfile
    real_mkstemp = tempfile.mkstemp

    def spy(*a, **kw):
        fd, path = real_mkstemp(*a, **kw)
        seen.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", spy)
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="i")
    save_state(s, tmp_path)
    save_state(s, tmp_path)
    assert len(seen) == 2 and seen[0] != seen[1]


def test_load_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_state(tmp_path)
