import pytest
from pathlib import Path
from kaidzen.candidate import Candidate, load_candidate

VALID_CONFIG = """\
domain: "тест"
rubric:
  novelty: "новизна"
  feasibility: "реализуемость"
  groundedness: "обоснованность"
  potential: "потенциал"
  clarity: "ясность"
researcher_focus: "аналоги"
analyzer_hints: ""
loop:
  max_iterations: 6
  plateau_threshold: 0.5
  assumptions_per_iteration: 3
models:
  analyzer: claude-sonnet-5
  researcher: claude-sonnet-5
  refiner: claude-sonnet-5
  judge: claude-sonnet-5
"""

PROMPTS = ["analyzer.md", "researcher.md", "refiner.md", "judge.md"]


def make_candidate(root: Path, config_text: str = VALID_CONFIG) -> Path:
    """Собирает валидного кандидата на диске; тесты портят его точечно."""
    d = root / "gen000-test"
    (d / "prompts").mkdir(parents=True)
    (d / "config.yaml").write_text(config_text, encoding="utf-8")
    for p in PROMPTS:
        (d / "prompts" / p).write_text(f"prompt {p}", encoding="utf-8")
    return d


def test_load_valid_candidate(tmp_path):
    c = load_candidate(make_candidate(tmp_path))
    assert c.candidate_id == "gen000-test"
    assert "groundedness" in c.config.rubric
    assert c.prompts["judge"] == "prompt judge.md"
    assert c.config.loop.max_iterations == 6


def test_rubric_must_have_groundedness(tmp_path):
    bad = VALID_CONFIG.replace("groundedness", "beauty")
    with pytest.raises(ValueError, match="groundedness"):
        load_candidate(make_candidate(tmp_path, bad))


def test_rubric_must_have_5_axes(tmp_path):
    bad = VALID_CONFIG.replace('  clarity: "ясность"\n', "")
    with pytest.raises(ValueError, match="5"):
        load_candidate(make_candidate(tmp_path, bad))


def test_models_must_cover_all_roles(tmp_path):
    bad = VALID_CONFIG.replace("  judge: claude-sonnet-5\n", "")
    with pytest.raises(ValueError, match="judge"):
        load_candidate(make_candidate(tmp_path, bad))


def test_missing_prompt_raises(tmp_path):
    d = make_candidate(tmp_path)
    (d / "prompts" / "judge.md").unlink()
    with pytest.raises(FileNotFoundError, match="judge"):
        load_candidate(d)


def test_missing_config_raises(tmp_path):
    d = make_candidate(tmp_path)
    (d / "config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        load_candidate(d)


def test_loop_defaults_when_section_absent(tmp_path):
    bare = VALID_CONFIG.split("loop:")[0] + """models:
  analyzer: m
  researcher: m
  refiner: m
  judge: m
"""
    c = load_candidate(make_candidate(tmp_path, bare))
    assert c.config.loop.max_iterations == 6
    assert c.config.loop.plateau_threshold == 0.5
    assert c.config.loop.assumptions_per_iteration == 3
