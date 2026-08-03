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
backends:
  subscription: { type: claude_agent_sdk }
roles:
  analyzer:   { backend: subscription, model: claude-sonnet-5 }
  researcher: { backend: subscription, model: claude-sonnet-5 }
  refiner:    { backend: subscription, model: claude-sonnet-5 }
  judge:      { backend: subscription, model: claude-sonnet-5 }
  reporter:   { backend: subscription, model: claude-sonnet-5 }
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


def role_line(role: str, backend: str = "subscription") -> str:
    """Строка роли из VALID_CONFIG — тесты подменяют её точечно."""
    pad = " " * (len("researcher") - len(role))
    return f"  {role}:{pad} {{ backend: {backend}, model: claude-sonnet-5 }}\n"


def test_roles_must_cover_all_of_them(tmp_path):
    bad = VALID_CONFIG.replace(role_line("judge"), "")
    with pytest.raises(ValueError, match="judge"):
        load_candidate(make_candidate(tmp_path, bad))


def test_roles_must_include_reporter(tmp_path):
    """Пятая роль: executive summary читает roles.reporter в конце прогона —
    без проверки при загрузке конфиг ломается только после оплаченного цикла."""
    bad = VALID_CONFIG.replace(role_line("reporter"), "")
    with pytest.raises(ValueError, match="reporter"):
        load_candidate(make_candidate(tmp_path, bad))


def test_unknown_role_rejected(tmp_path):
    bad = VALID_CONFIG + role_line("designer")
    with pytest.raises(ValueError, match="designer"):
        load_candidate(make_candidate(tmp_path, bad))


def test_role_backend_must_be_declared(tmp_path):
    bad = VALID_CONFIG.replace(role_line("refiner"),
                               role_line("refiner", "openai"))
    with pytest.raises(ValueError, match="openai"):
        load_candidate(make_candidate(tmp_path, bad))


def test_unknown_backend_type_rejected(tmp_path):
    bad = VALID_CONFIG.replace("type: claude_agent_sdk", "type: telepathy")
    with pytest.raises(ValueError, match="telepathy"):
        load_candidate(make_candidate(tmp_path, bad))


def test_researcher_on_backend_without_web_search_rejected(tmp_path):
    """Главная проверка §3.3: без поиска Researcher вернёт ссылки из памяти."""
    bad = (VALID_CONFIG
           .replace("  subscription: { type: claude_agent_sdk }",
                    "  subscription: { type: claude_agent_sdk }\n"
                    "  deepseek: { type: openai_compat, "
                    "api_key_env: DEEPSEEK_API_KEY }")
           .replace(role_line("researcher"),
                    role_line("researcher", "deepseek")))

    with pytest.raises(ValueError) as exc:
        load_candidate(make_candidate(tmp_path, bad))

    message = str(exc.value)
    assert "deepseek" in message          # назван сам бэкенд
    assert "поиск" in message             # и причина отказа


def test_non_searching_backend_is_fine_for_other_roles(tmp_path):
    """openai_compat не умеет искать, но Judge поиск и не нужен."""
    ok = (VALID_CONFIG
          .replace("  subscription: { type: claude_agent_sdk }",
                   "  subscription: { type: claude_agent_sdk }\n"
                   "  deepseek: { type: openai_compat, "
                   "api_key_env: DEEPSEEK_API_KEY }")
          .replace(role_line("judge"), role_line("judge", "deepseek")))

    candidate = load_candidate(make_candidate(tmp_path, ok))

    assert candidate.config.roles["judge"].backend == "deepseek"


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


def test_unknown_top_level_key_rejected(tmp_path):
    """Опечатка в ключе не должна молча включать дефолты."""
    bad = VALID_CONFIG.replace("loop:", "lop:")
    with pytest.raises(ValueError, match="lop"):
        load_candidate(make_candidate(tmp_path, bad))


def test_unknown_loop_key_rejected(tmp_path):
    bad = VALID_CONFIG.replace("  max_iterations: 6", "  max_iteration: 6")
    with pytest.raises(ValueError, match="max_iteration"):
        load_candidate(make_candidate(tmp_path, bad))


@pytest.mark.parametrize("line,broken", [
    ("  max_iterations: 6", "  max_iterations: 0"),
    ("  max_iterations: 6", "  max_iterations: 21"),
    ("  plateau_threshold: 0.5", "  plateau_threshold: -3"),
    ("  plateau_threshold: 0.5", "  plateau_threshold: 11"),
    ("  assumptions_per_iteration: 3", "  assumptions_per_iteration: 0"),
    ("  assumptions_per_iteration: 3", "  assumptions_per_iteration: 11"),
])
def test_loop_params_are_bounded(tmp_path, line, broken):
    with pytest.raises(ValueError):
        load_candidate(make_candidate(tmp_path, VALID_CONFIG.replace(line, broken)))


def test_empty_prompt_file_rejected(tmp_path):
    d = make_candidate(tmp_path)
    (d / "prompts" / "judge.md").write_text("   \n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="judge"):
        load_candidate(d)


def test_blank_model_name_rejected(tmp_path):
    bad = VALID_CONFIG.replace(role_line("judge"),
                               "  judge:      { backend: subscription, "
                               'model: "" }\n')
    with pytest.raises(ValueError, match="model"):
        load_candidate(make_candidate(tmp_path, bad))


def test_blank_domain_rejected(tmp_path):
    bad = VALID_CONFIG.replace('domain: "тест"', 'domain: "  "')
    with pytest.raises(ValueError, match="domain"):
        load_candidate(make_candidate(tmp_path, bad))


def test_blank_rubric_axis_description_rejected(tmp_path):
    bad = VALID_CONFIG.replace('  clarity: "ясность"', '  clarity: ""')
    with pytest.raises(ValueError, match="clarity"):
        load_candidate(make_candidate(tmp_path, bad))


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "просто строка\n"])
def test_non_mapping_config_rejected(tmp_path, text):
    with pytest.raises(ValueError, match="config.yaml"):
        load_candidate(make_candidate(tmp_path, text))


def test_loop_defaults_when_section_absent(tmp_path):
    bare = VALID_CONFIG.split("loop:")[0] + VALID_CONFIG.split("backends:")[1]
    bare = bare.replace("  subscription", "backends:\n  subscription", 1)
    c = load_candidate(make_candidate(tmp_path, bare))
    assert c.config.loop.max_iterations == 6
    assert c.config.loop.plateau_threshold == 0.5
    assert c.config.loop.assumptions_per_iteration == 3


# --- кандидаты, поставляемые в репозитории -----------------------------------

REPO_CANDIDATES = Path(__file__).parent.parent / "candidates"
SHIPPED_DOMAINS = ["generic", "business", "games"]
# заглушка вида "prompt judge.md" проходит валидацию, но промптом не является
MIN_PROMPT_LENGTH = 200


@pytest.fixture(params=SHIPPED_DOMAINS)
def shipped(request) -> Candidate:
    return load_candidate(REPO_CANDIDATES / f"gen000-{request.param}")


def test_shipped_candidate_loads(shipped):
    assert shipped.candidate_id.startswith("gen000-")
    assert shipped.config.domain.strip()


def test_shipped_candidate_rubric(shipped):
    assert "groundedness" in shipped.config.rubric
    assert len(shipped.config.rubric) == 5


@pytest.mark.parametrize("domain", SHIPPED_DOMAINS)
def test_champion_file_points_at_loadable_candidate(domain):
    """Указатель обязан вести на загружаемого кандидата — но не обязательно
    на исходного: смена чемпиона эволюцией это и есть работа мета-лупа."""
    champ = (REPO_CANDIDATES / f"CHAMPION-{domain}").read_text(
        encoding="utf-8").strip()
    assert load_candidate(REPO_CANDIDATES / champ).candidate_id == champ


def test_shipped_prompts_are_not_stubs(shipped):
    for role, text in shipped.prompts.items():
        assert len(text) > MIN_PROMPT_LENGTH, f"{shipped.candidate_id}/{role}"


def test_shipped_judge_prompt_states_total_is_sum(shipped):
    """Без этого правила JudgeResult валит валидацию и прогон жжёт ретраи."""
    judge = shipped.prompts["judge"]
    assert "total" in judge
    assert "сумм" in judge.lower()


def test_shipped_researcher_prompt_forbids_invented_urls(shipped):
    researcher = shipped.prompts["researcher"].lower()
    assert "придумывать" in researcher
    assert "запрещено" in researcher


def test_empty_backends_section_rejected(tmp_path):
    bad = VALID_CONFIG.replace("backends:\n  subscription: { type: claude_agent_sdk }\n",
                               "backends: {}\n")
    with pytest.raises(ValueError, match="backends"):
        load_candidate(make_candidate(tmp_path, bad))
