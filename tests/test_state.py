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


def test_judge_verdict_enum():
    with pytest.raises(ValidationError):
        JudgeResult(scores={"clarity": 5}, total=5.0,
                    delta_vs_previous=0.0, critique=[], verdict="maybe")


def test_runstate_roundtrip():
    s = RunState(run_id="r1", candidate_id="gen000-generic",
                 config={}, original_idea="idea")
    s2 = RunState.model_validate_json(s.model_dump_json())
    assert s2.iteration == 0 and s2.versions == []


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


def test_load_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_state(tmp_path)
