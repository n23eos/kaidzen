# Kaidzen Level 1 (Inner Loop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Рабочий CLI-инструмент: `kaidzen run idea.md` шлифует идею циклом Analyzer→(Researcher→Refiner→Judge)* с веб-grounding и выдаёт report.md.

**Architecture:** Чистый Python-оркестратор; 4 роли = 4 вызова Claude API со structured output (tool-use + schema); state — атомарно записываемый JSON с resume; кандидат (config.yaml + промпты) — сменная папка. Мета-луп (Уровень 2) — отдельный план после E2E этого.

**Tech Stack:** Python 3.12+, `anthropic` SDK (server-side web_search tool), `pydantic` v2, `pyyaml`, `pytest`.

**Spec:** `docs/specs/2026-08-03-idea-refinement-loop-tz.md` (§1–§3, §5–§10, этапы 1–5 из §11).

---

## File Structure

```
kaidzen/
├── pyproject.toml
├── kaidzen/
│   ├── __init__.py
│   ├── __main__.py        # CLI: run / resume / report
│   ├── state.py           # pydantic-схемы + атомарная запись/чтение
│   ├── candidate.py       # загрузка и валидация кандидата (config.yaml + prompts/)
│   ├── llm.py             # обёртка SDK: structured output, retry, usage
│   ├── orchestrator.py    # цикл, стоп-критерии, выбор допущений
│   ├── report.py          # report.md из state
│   └── roles/
│       ├── __init__.py
│       ├── analyzer.py
│       ├── researcher.py
│       ├── refiner.py
│       └── judge.py
├── candidates/
│   ├── CHAMPION-generic   # текстовый файл с id champion-кандидата
│   ├── gen000-generic/{config.yaml, prompts/*.md, meta.json}
│   ├── gen000-business/{...}
│   └── gen000-games/{...}
├── tests/
│   ├── conftest.py
│   ├── test_state.py
│   ├── test_candidate.py
│   ├── test_llm.py
│   ├── test_orchestrator.py
│   ├── test_roles.py
│   └── test_report.py
└── runs/                  # gitignore
```

---

### Task 1: Каркас проекта

**Files:**
- Create: `pyproject.toml`, `kaidzen/__init__.py`, `.gitignore`, `tests/conftest.py`

- [ ] **Step 1: git init + скелет**

```bash
cd kaidzen
git init
mkdir -p kaidzen/roles tests candidates runs
touch kaidzen/__init__.py kaidzen/roles/__init__.py
```

- [ ] **Step 2: pyproject.toml**

```toml
[project]
name = "kaidzen"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["anthropic>=0.40", "pydantic>=2.7", "pyyaml>=6.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["kaidzen*"]
```

- [ ] **Step 3: .gitignore**

```
runs/
evolve/
__pycache__/
*.egg-info/
.venv/
.coverage
```

- [ ] **Step 4: venv + установка**

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Expected: успешная установка, `.venv/bin/pytest --collect-only` отрабатывает (0 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: project skeleton"
```

---

### Task 2: Схемы state (pydantic)

**Files:**
- Create: `kaidzen/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Failing test — схемы и валидация**

```python
# tests/test_state.py
import pytest
from pydantic import ValidationError
from kaidzen.state import Assumption, Fact, JudgeResult, Version, RunState


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
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/pytest tests/test_state.py -v`
Expected: FAIL, `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Реализация схем**

```python
# kaidzen/state.py
"""Схемы состояния прогона и атомарная запись state.json."""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

Criticality = Literal["high", "medium", "low"]
AssumptionStatus = Literal["unverified", "confirmed", "refuted", "partial", "untestable"]


class Fact(BaseModel):
    claim: str
    source_url: str
    source_title: str = ""
    date: str = ""


class Assumption(BaseModel):
    id: str
    text: str
    criticality: Criticality
    status: AssumptionStatus = "unverified"
    facts: list[Fact] = Field(default_factory=list)


class Analysis(BaseModel):
    problem: str
    audience: str
    mechanism: str
    unknowns: list[str] = Field(default_factory=list)


class ChangelogEntry(BaseModel):
    change: str
    reason: str
    grounded_in: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    scores: dict[str, float]
    total: float
    delta_vs_previous: float
    critique: list[str]
    verdict: Literal["continue", "rollback"]


class Version(BaseModel):
    n: int
    idea_text: str
    changelog: list[ChangelogEntry] = Field(default_factory=list)
    judge: Optional[JudgeResult] = None
    rolled_back: bool = False


class ApiUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0


class RunState(BaseModel):
    run_id: str
    candidate_id: str
    config: dict[str, Any]
    original_idea: str
    analysis: Optional[Analysis] = None
    assumptions: list[Assumption] = Field(default_factory=list)
    versions: list[Version] = Field(default_factory=list)
    iteration: int = 0
    rollbacks: int = 0
    consecutive_rollbacks: int = 0
    low_delta_streak: int = 0
    stop_reason: Optional[str] = None
    last_completed_step: Optional[str] = None
    api_usage: ApiUsage = Field(default_factory=ApiUsage)

    def current_version(self) -> Optional[Version]:
        active = [v for v in self.versions if not v.rolled_back]
        return active[-1] if active else None

    def current_idea_text(self) -> str:
        v = self.current_version()
        return v.idea_text if v else self.original_idea
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/pytest tests/test_state.py -v` → PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: state schemas"
```

---

### Task 3: Атомарная запись и resume-чтение state

**Files:**
- Modify: `kaidzen/state.py` (добавить функции в конец)
- Test: `tests/test_state.py` (добавить)

- [ ] **Step 1: Failing tests**

```python
# tests/test_state.py — добавить
from pathlib import Path
from kaidzen.state import save_state, load_state


def test_save_load_roundtrip(tmp_path: Path):
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="i")
    save_state(s, tmp_path)
    loaded = load_state(tmp_path)
    assert loaded == s


def test_save_is_atomic_no_tmp_leftover(tmp_path: Path):
    s = RunState(run_id="r1", candidate_id="c", config={}, original_idea="i")
    save_state(s, tmp_path)
    assert (tmp_path / "state.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_load_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_state(tmp_path)
```

- [ ] **Step 2: Run — FAIL** (`ImportError: save_state`)

- [ ] **Step 3: Реализация**

```python
# kaidzen/state.py — добавить в конец
import os
from pathlib import Path


def save_state(state: RunState, run_dir: Path) -> None:
    """Атомарно: пишем во временный файл, затем rename."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "state.json")


def load_state(run_dir: Path) -> RunState:
    path = run_dir / "state.json"
    if not path.exists():
        raise FileNotFoundError(f"state.json не найден в {run_dir}")
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))
```

(`test_save_is_atomic_no_tmp_leftover` использует glob `*.tmp` — имя временного файла `state.json.tmp` под него попадает; после `os.replace` файл исчезает.)

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: atomic state save/load"`

---

### Task 4: Кандидат — конфиг и промпты

**Files:**
- Create: `kaidzen/candidate.py`
- Test: `tests/test_candidate.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_candidate.py
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


def test_rubric_must_have_groundedness(tmp_path):
    bad = VALID_CONFIG.replace("groundedness", "beauty")
    with pytest.raises(ValueError, match="groundedness"):
        load_candidate(make_candidate(tmp_path, bad))


def test_rubric_must_have_5_axes(tmp_path):
    bad = VALID_CONFIG.replace('  clarity: "ясность"\n', "")
    with pytest.raises(ValueError, match="5"):
        load_candidate(make_candidate(tmp_path, bad))


def test_missing_prompt_raises(tmp_path):
    d = make_candidate(tmp_path)
    (d / "prompts" / "judge.md").unlink()
    with pytest.raises(FileNotFoundError, match="judge"):
        load_candidate(d)
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

```python
# kaidzen/candidate.py
"""Кандидат = папка: config.yaml + prompts/{analyzer,researcher,refiner,judge}.md."""
from __future__ import annotations

from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator

ROLES = ("analyzer", "researcher", "refiner", "judge")
RUBRIC_AXES = 5


class LoopConfig(BaseModel):
    max_iterations: int = 6
    plateau_threshold: float = 0.5
    assumptions_per_iteration: int = 3


class CandidateConfig(BaseModel):
    domain: str
    rubric: dict[str, str]
    researcher_focus: str = ""
    analyzer_hints: str = ""
    loop: LoopConfig = Field(default_factory=LoopConfig)
    models: dict[str, str]

    @field_validator("rubric")
    @classmethod
    def check_rubric(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) != RUBRIC_AXES:
            raise ValueError(f"рубрика должна содержать ровно {RUBRIC_AXES} осей, получено {len(v)}")
        if "groundedness" not in v:
            raise ValueError("ось groundedness обязательна в рубрике")
        return v

    @field_validator("models")
    @classmethod
    def check_models(cls, v: dict[str, str]) -> dict[str, str]:
        missing = [r for r in ROLES if r not in v]
        if missing:
            raise ValueError(f"models: не заданы модели для ролей {missing}")
        return v


class Candidate(BaseModel):
    candidate_id: str
    config: CandidateConfig
    prompts: dict[str, str]


def load_candidate(path: Path) -> Candidate:
    config_path = path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"нет config.yaml в {path}")
    config = CandidateConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8")))
    prompts: dict[str, str] = {}
    for role in ROLES:
        p = path / "prompts" / f"{role}.md"
        if not p.exists():
            raise FileNotFoundError(f"нет промпта роли {role}: {p}")
        prompts[role] = p.read_text(encoding="utf-8")
    return Candidate(candidate_id=path.name, config=config, prompts=prompts)
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `git commit -am "feat: candidate loading and validation"` (плюс `git add` новых файлов)

---

### Task 5: LLM-обёртка — structured output + retry

**Files:**
- Create: `kaidzen/llm.py`
- Test: `tests/test_llm.py`

Принцип: заставляем модель вызвать tool `submit` со схемой ответа (`tool_choice` type=tool, когда нет web_search; с web_search — инструкция + цикл до появления tool_use `submit`). Валидация pydantic; при ошибке — 1 повтор с текстом ошибки. Все вызовы через `LLMClient`, который в тестах подменяется фейком.

- [ ] **Step 1: Failing tests**

```python
# tests/test_llm.py
import pytest
from pydantic import BaseModel
from kaidzen.llm import LLMClient, SchemaValidationFailure


class Out(BaseModel):
    answer: str
    score: int


class FakeBlock:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class FakeResponse:
    def __init__(self, blocks, in_tok=10, out_tok=5):
        self.content = blocks
        self.usage = type("U", (), {"input_tokens": in_tok, "output_tokens": out_tok,
                                    "server_tool_use": None})()


class FakeAnthropicMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def make_client(responses):
    client = LLMClient(api_key="test")
    fake = FakeAnthropicMessages(responses)
    client._client = type("C", (), {"messages": fake})()
    return client, fake


def good_response():
    return FakeResponse([FakeBlock("tool_use", name="submit",
                                   input={"answer": "ok", "score": 7})])


def bad_then_good():
    return [FakeResponse([FakeBlock("tool_use", name="submit",
                                    input={"answer": "ok"})]),  # нет score
            good_response()]


def test_structured_returns_validated_model():
    client, _ = make_client([good_response()])
    out = client.structured(model="m", system="s", user="u",
                            schema=Out, temperature=0.1)
    assert out == Out(answer="ok", score=7)


def test_retry_once_on_schema_error_then_success():
    client, fake = make_client(bad_then_good())
    out = client.structured(model="m", system="s", user="u",
                            schema=Out, temperature=0.1)
    assert out.score == 7
    assert len(fake.calls) == 2
    # повторный вызов содержит текст ошибки валидации
    assert "score" in str(fake.calls[1]["messages"])


def test_two_schema_failures_raise():
    client, _ = make_client([bad_then_good()[0], bad_then_good()[0]])
    with pytest.raises(SchemaValidationFailure):
        client.structured(model="m", system="s", user="u",
                          schema=Out, temperature=0.1)


def test_usage_accumulates():
    client, _ = make_client([good_response(), good_response()])
    client.structured(model="m", system="s", user="u", schema=Out, temperature=0)
    client.structured(model="m", system="s", user="u", schema=Out, temperature=0)
    assert client.usage.input_tokens == 20
    assert client.usage.output_tokens == 10
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

```python
# kaidzen/llm.py
"""Обёртка Anthropic SDK: structured output через tool-use, retry, учёт usage."""
from __future__ import annotations

from typing import Type, TypeVar
import anthropic
from pydantic import BaseModel, ValidationError

from kaidzen.state import ApiUsage

T = TypeVar("T", bound=BaseModel)
MAX_SCHEMA_RETRIES = 1  # один повтор с текстом ошибки
SUBMIT_TOOL = "submit"


class SchemaValidationFailure(Exception):
    pass


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.usage = ApiUsage()

    def _submit_tool(self, schema: Type[BaseModel]) -> dict:
        return {
            "name": SUBMIT_TOOL,
            "description": "Отправить финальный структурированный ответ.",
            "input_schema": schema.model_json_schema(),
        }

    def _record_usage(self, response) -> None:
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens
        stu = getattr(response.usage, "server_tool_use", None)
        if stu is not None:
            self.usage.web_searches += getattr(stu, "web_search_requests", 0) or 0

    def _extract_submit(self, response) -> dict | None:
        for block in response.content:
            if block.type == "tool_use" and block.name == SUBMIT_TOOL:
                return block.input
        return None

    def structured(self, *, model: str, system: str, user: str,
                   schema: Type[T], temperature: float,
                   web_search: bool = False, max_searches: int = 8,
                   max_tokens: int = 4096) -> T:
        tools: list[dict] = [self._submit_tool(schema)]
        kwargs: dict = {}
        if web_search:
            tools.append({"type": "web_search_20250305", "name": "web_search",
                          "max_uses": max_searches})
            # с web_search нельзя форсировать tool_choice — модель должна
            # сначала искать; полагаемся на инструкцию в системном промпте
        else:
            kwargs["tool_choice"] = {"type": "tool", "name": SUBMIT_TOOL}

        messages = [{"role": "user", "content": user}]
        last_error = ""
        for attempt in range(MAX_SCHEMA_RETRIES + 1):
            if last_error:
                messages = messages + [{
                    "role": "user",
                    "content": f"Твой прошлый ответ не прошёл валидацию схемы: {last_error}. "
                               f"Вызови tool '{SUBMIT_TOOL}' ещё раз с исправленными полями."}]
            response = self._client.messages.create(
                model=model, system=system, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                tools=tools, **kwargs)
            self._record_usage(response)
            payload = self._extract_submit(response)
            if payload is None:
                last_error = f"ответ не содержит вызова tool '{SUBMIT_TOOL}'"
                continue
            try:
                return schema.model_validate(payload)
            except ValidationError as e:
                last_error = str(e)
        raise SchemaValidationFailure(last_error)
```

Примечание: сетевые ретраи (rate limit, overloaded) SDK делает сам (`max_retries` по умолчанию 2); поверх этого повтор шага целиком делает оркестратор (Task 8).

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: LLM wrapper with structured output and schema retry"`

---

### Task 6: Роли — контракты и вызовы

**Files:**
- Create: `kaidzen/roles/analyzer.py`, `researcher.py`, `refiner.py`, `judge.py`
- Test: `tests/test_roles.py`, `tests/conftest.py`

Каждая роль: собственная выходная pydantic-схема + функция `run_<role>(llm, candidate, ...) -> схема`. Роли не знают про state целиком — получают только нужные куски.

- [ ] **Step 1: conftest — фейковый LLM**

```python
# tests/conftest.py
import pytest
from pydantic import BaseModel


class FakeLLM:
    """Отдаёт заранее заданные объекты, записывает вызовы."""
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def candidate(tmp_path):
    from tests.test_candidate import make_candidate
    from kaidzen.candidate import load_candidate
    return load_candidate(make_candidate(tmp_path))
```

- [ ] **Step 2: Failing tests**

```python
# tests/test_roles.py
from kaidzen.roles.analyzer import AnalyzerOutput, run_analyzer
from kaidzen.roles.researcher import ResearchFinding, ResearcherOutput, run_researcher
from kaidzen.roles.refiner import RefinerOutput, run_refiner
from kaidzen.roles.judge import run_judge
from kaidzen.state import Assumption, Fact, JudgeResult, ChangelogEntry
from tests.conftest import FakeLLM


def _analyzer_out():
    return AnalyzerOutput(
        problem="p", audience="a", mechanism="m", unknowns=["u"],
        assumptions=[Assumption(id="A1", text="t", criticality="high")])


def test_analyzer_passes_idea_and_hints(candidate):
    llm = FakeLLM([_analyzer_out()])
    out = run_analyzer(llm, candidate, idea_text="моя идея")
    assert out.assumptions[0].id == "A1"
    call = llm.calls[0]
    assert "моя идея" in call["user"]
    assert call["system"] == candidate.prompts["analyzer"]
    assert call["temperature"] == 0.3


def test_researcher_uses_web_search(candidate):
    finding = ResearchFinding(assumption_id="A1", verdict="confirmed",
                              facts=[Fact(claim="c", source_url="http://x")], notes="")
    llm = FakeLLM([ResearcherOutput(findings=[finding])])
    out = run_researcher(llm, candidate, idea_text="i",
                         assumptions=[Assumption(id="A1", text="t", criticality="high")])
    assert out.findings[0].verdict == "confirmed"
    assert llm.calls[0]["web_search"] is True
    assert llm.calls[0]["temperature"] == 0.2


def test_refiner_gets_critique_and_findings(candidate):
    llm = FakeLLM([RefinerOutput(idea_text="v2", changelog=[
        ChangelogEntry(change="c", reason="r", grounded_in=["A1"])])])
    out = run_refiner(llm, candidate, idea_text="v1",
                      findings_json="[...]", critique=["слабое место"])
    assert out.idea_text == "v2"
    assert "слабое место" in llm.calls[0]["user"]
    assert llm.calls[0]["temperature"] == 0.7


def test_judge_never_sees_changelog(candidate):
    jr = JudgeResult(scores={k: 5 for k in candidate.config.rubric}, total=25,
                     delta_vs_previous=0, critique=[], verdict="continue")
    llm = FakeLLM([jr])
    run_judge(llm, candidate, new_idea="v2", previous_idea="v1",
              assumptions=[Assumption(id="A1", text="t", criticality="high")])
    user_msg = llm.calls[0]["user"]
    assert "changelog" not in user_msg.lower()
    assert llm.calls[0]["temperature"] == 0.1
```

- [ ] **Step 3: Run — FAIL**

- [ ] **Step 4: Реализация ролей**

```python
# kaidzen/roles/analyzer.py
from pydantic import BaseModel, Field
from kaidzen.candidate import Candidate
from kaidzen.state import Assumption

TEMPERATURE = 0.3


class AnalyzerOutput(BaseModel):
    problem: str
    audience: str
    mechanism: str
    assumptions: list[Assumption]
    unknowns: list[str] = Field(default_factory=list)


def run_analyzer(llm, candidate: Candidate, *, idea_text: str) -> AnalyzerOutput:
    user = (f"Идея для декомпозиции:\n\n{idea_text}\n\n"
            f"Домен: {candidate.config.domain}\n"
            f"Подсказки: {candidate.config.analyzer_hints}")
    return llm.structured(model=candidate.config.models["analyzer"],
                          system=candidate.prompts["analyzer"], user=user,
                          schema=AnalyzerOutput, temperature=TEMPERATURE)
```

```python
# kaidzen/roles/researcher.py
from typing import Literal
from pydantic import BaseModel, Field
from kaidzen.candidate import Candidate
from kaidzen.state import Assumption, Fact

TEMPERATURE = 0.2
MAX_SEARCHES_PER_CALL = 8


class ResearchFinding(BaseModel):
    assumption_id: str
    verdict: Literal["confirmed", "refuted", "partial", "untestable"]
    facts: list[Fact] = Field(default_factory=list)
    notes: str = ""


class ResearcherOutput(BaseModel):
    findings: list[ResearchFinding]


def run_researcher(llm, candidate: Candidate, *, idea_text: str,
                   assumptions: list[Assumption]) -> ResearcherOutput:
    listing = "\n".join(f"- {a.id} [{a.criticality}]: {a.text}" for a in assumptions)
    user = (f"Текущая версия идеи:\n\n{idea_text}\n\n"
            f"Проверь эти допущения веб-поиском:\n{listing}\n\n"
            f"Фокус поиска: {candidate.config.researcher_focus}")
    return llm.structured(model=candidate.config.models["researcher"],
                          system=candidate.prompts["researcher"], user=user,
                          schema=ResearcherOutput, temperature=TEMPERATURE,
                          web_search=True, max_searches=MAX_SEARCHES_PER_CALL)
```

```python
# kaidzen/roles/refiner.py
from pydantic import BaseModel, Field
from kaidzen.candidate import Candidate
from kaidzen.state import ChangelogEntry

TEMPERATURE = 0.7


class RefinerOutput(BaseModel):
    idea_text: str
    changelog: list[ChangelogEntry] = Field(default_factory=list)


def run_refiner(llm, candidate: Candidate, *, idea_text: str,
                findings_json: str, critique: list[str]) -> RefinerOutput:
    crit = "\n".join(f"- {c}" for c in critique) or "(первая итерация, критики нет)"
    user = (f"Текущая версия идеи:\n\n{idea_text}\n\n"
            f"Свежие находки Researcher (JSON):\n{findings_json}\n\n"
            f"Критика Judge с прошлой итерации:\n{crit}")
    return llm.structured(model=candidate.config.models["refiner"],
                          system=candidate.prompts["refiner"], user=user,
                          schema=RefinerOutput, temperature=TEMPERATURE)
```

```python
# kaidzen/roles/judge.py
from kaidzen.candidate import Candidate
from kaidzen.state import Assumption, JudgeResult

TEMPERATURE = 0.1


def run_judge(llm, candidate: Candidate, *, new_idea: str, previous_idea: str,
              assumptions: list[Assumption]) -> JudgeResult:
    rubric = "\n".join(f"- {axis}: {desc}" for axis, desc in candidate.config.rubric.items())
    reg = "\n".join(f"- {a.id} [{a.status}]: {a.text}" for a in assumptions)
    # ВАЖНО: changelog Refiner'а сюда не передаётся — Judge оценивает
    # результат, а не нарратив об изменениях (ТЗ §2.1).
    user = (f"Рубрика (каждая ось 0–10):\n{rubric}\n\n"
            f"Реестр допущений:\n{reg}\n\n"
            f"ПРЕДЫДУЩАЯ версия идеи:\n{previous_idea}\n\n"
            f"НОВАЯ версия идеи:\n{new_idea}")
    return llm.structured(model=candidate.config.models["judge"],
                          system=candidate.prompts["judge"], user=user,
                          schema=JudgeResult, temperature=TEMPERATURE)
```

- [ ] **Step 5: Run — PASS** (`.venv/bin/pytest tests/test_roles.py -v`)

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: four role contracts and calls"`

---

### Task 7: Чистая логика оркестратора — стоп-критерии и выбор допущений

**Files:**
- Create: `kaidzen/orchestrator.py` (пока только чистые функции)
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_orchestrator.py
from kaidzen.state import Assumption, JudgeResult, RunState, Version
from kaidzen.candidate import LoopConfig
from kaidzen.orchestrator import check_stop, select_assumptions, apply_judge_verdict


def make_state(**kw):
    base = dict(run_id="r", candidate_id="c", config={}, original_idea="i")
    base.update(kw)
    return RunState(**base)


def jr(total, delta, verdict="continue"):
    return JudgeResult(scores={}, total=total, delta_vs_previous=delta,
                       critique=[], verdict=verdict)


LOOP = LoopConfig(max_iterations=6, plateau_threshold=0.5, assumptions_per_iteration=3)


def test_stop_on_max_iterations():
    s = make_state(iteration=6)
    assert check_stop(s, LOOP) == "max_iterations"


def test_stop_on_plateau_two_low_deltas():
    s = make_state(iteration=3, low_delta_streak=2)
    assert check_stop(s, LOOP) == "plateau"


def test_stop_when_high_assumptions_exhausted():
    s = make_state(iteration=2, assumptions=[
        Assumption(id="A1", text="t", criticality="high", status="confirmed"),
        Assumption(id="A2", text="t", criticality="high", status="untestable"),
        Assumption(id="A3", text="t", criticality="low", status="unverified")])
    assert check_stop(s, LOOP) == "assumptions_exhausted"


def test_stop_on_double_rollback():
    s = make_state(iteration=3, consecutive_rollbacks=2,
                   assumptions=[Assumption(id="A1", text="t", criticality="high")])
    assert check_stop(s, LOOP) == "degrading"


def test_no_stop_mid_run():
    s = make_state(iteration=2, low_delta_streak=1,
                   assumptions=[Assumption(id="A1", text="t", criticality="high")])
    assert check_stop(s, LOOP) is None


def test_select_assumptions_by_criticality_and_status():
    a = [Assumption(id="A1", text="t", criticality="low"),
         Assumption(id="A2", text="t", criticality="high"),
         Assumption(id="A3", text="t", criticality="high", status="confirmed"),
         Assumption(id="A4", text="t", criticality="medium"),
         Assumption(id="A5", text="t", criticality="high", status="untestable")]
    picked = select_assumptions(a, limit=2)
    assert [x.id for x in picked] == ["A2", "A4"]  # только unverified, high прежде medium


def test_apply_verdict_rollback_marks_version():
    s = make_state(versions=[
        Version(n=1, idea_text="v1", judge=jr(20, 0)),
        Version(n=2, idea_text="v2")])
    apply_judge_verdict(s, jr(15, -5, "rollback"), LOOP)
    assert s.versions[-1].rolled_back is True
    assert s.consecutive_rollbacks == 1
    assert s.current_idea_text() == "v1"


def test_apply_verdict_continue_updates_streak():
    s = make_state(versions=[Version(n=1, idea_text="v1")])
    apply_judge_verdict(s, jr(20, 0.3), LOOP)   # ниже порога 0.5
    assert s.low_delta_streak == 1
    apply_judge_verdict(s, jr(21, 1.0), LOOP)   # выше порога — сброс
    assert s.low_delta_streak == 0
    assert s.consecutive_rollbacks == 0
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

```python
# kaidzen/orchestrator.py
"""Цикл Уровня 1: стоп-критерии, выбор допущений, применение вердикта."""
from __future__ import annotations

from typing import Optional
from kaidzen.candidate import LoopConfig
from kaidzen.state import Assumption, JudgeResult, RunState

ROLLBACK_LIMIT = 2       # два отката подряд = деградация
PLATEAU_STREAK = 2       # два низких прироста подряд = плато
CRITICALITY_ORDER = {"high": 0, "medium": 1, "low": 2}
CLOSED = {"confirmed", "refuted", "untestable"}


def check_stop(state: RunState, loop: LoopConfig) -> Optional[str]:
    if state.iteration >= loop.max_iterations:
        return "max_iterations"
    if state.consecutive_rollbacks >= ROLLBACK_LIMIT:
        return "degrading"
    if state.low_delta_streak >= PLATEAU_STREAK:
        return "plateau"
    high = [a for a in state.assumptions if a.criticality == "high"]
    if high and all(a.status in CLOSED for a in high):
        return "assumptions_exhausted"
    return None


def select_assumptions(assumptions: list[Assumption], limit: int) -> list[Assumption]:
    open_ones = [a for a in assumptions if a.status == "unverified"]
    open_ones.sort(key=lambda a: CRITICALITY_ORDER[a.criticality])
    return open_ones[:limit]


def apply_judge_verdict(state: RunState, judge: JudgeResult, loop: LoopConfig) -> None:
    latest = state.versions[-1]
    if judge.verdict == "rollback":
        latest.rolled_back = True
        state.rollbacks += 1
        state.consecutive_rollbacks += 1
        return
    latest.judge = judge
    state.consecutive_rollbacks = 0
    if judge.delta_vs_previous < loop.plateau_threshold:
        state.low_delta_streak += 1
    else:
        state.low_delta_streak = 0
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: orchestrator pure logic (stop criteria, selection, verdict)"`

---

### Task 8: Оркестратор — полный цикл с resume

**Files:**
- Modify: `kaidzen/orchestrator.py` (добавить `run_pipeline`)
- Test: `tests/test_orchestrator.py` (добавить)

Шаги пайплайна пишут state после каждого завершённого шага (`last_completed_step`: `analyzer` → `researcher` → `refiner` → `judge` → снова `researcher`...). Resume: `run_pipeline` смотрит `last_completed_step` и продолжает со следующего. Повтор упавшего шага: 3 попытки с паузой, потом исключение (state уже на диске).

- [ ] **Step 1: Failing tests**

```python
# tests/test_orchestrator.py — добавить
from pathlib import Path
from kaidzen.orchestrator import run_pipeline
from kaidzen.roles.analyzer import AnalyzerOutput
from kaidzen.roles.researcher import ResearchFinding, ResearcherOutput
from kaidzen.roles.refiner import RefinerOutput
from kaidzen.state import ChangelogEntry, Fact, load_state
from tests.conftest import FakeLLM


def full_run_outputs(rubric_axes):
    """Выходы ролей: analyzer + 1 полная итерация; после неё единственное
    high-допущение закрыто → стоп по assumptions_exhausted."""
    scores = {k: 5.0 for k in rubric_axes}
    return [
        AnalyzerOutput(problem="p", audience="a", mechanism="m", assumptions=[
            Assumption(id="A1", text="t", criticality="high")]),
        ResearcherOutput(findings=[ResearchFinding(
            assumption_id="A1", verdict="confirmed",
            facts=[Fact(claim="c", source_url="http://x")])]),
        RefinerOutput(idea_text="v1 better", changelog=[
            ChangelogEntry(change="c", reason="r", grounded_in=["A1"])]),
        JudgeResult(scores=scores, total=25, delta_vs_previous=2.0,
                    critique=["weak spot"], verdict="continue"),
    ]


def test_full_pipeline_stops_on_exhausted(tmp_path, candidate):
    llm = FakeLLM(full_run_outputs(candidate.config.rubric))
    state = run_pipeline(llm, candidate, idea_text="raw idea",
                         run_dir=tmp_path / "run1")
    assert state.stop_reason == "assumptions_exhausted"
    assert state.assumptions[0].status == "confirmed"
    assert state.assumptions[0].facts[0].source_url == "http://x"
    assert state.versions[-1].judge.total == 25
    # state лежит на диске и равен возвращённому
    assert load_state(tmp_path / "run1") == state


def test_resume_skips_completed_steps(tmp_path, candidate):
    outputs = full_run_outputs(candidate.config.rubric)
    # первый запуск падает после researcher (refiner бросает исключение)
    crash = FakeLLM(outputs[:2] + [RuntimeError("boom")] * 3)  # 3 попытки шага
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        run_pipeline(crash, candidate, idea_text="raw", run_dir=tmp_path / "r")
    saved = load_state(tmp_path / "r")
    assert saved.last_completed_step == "researcher"
    # resume: нужны только refiner и judge
    llm2 = FakeLLM(outputs[2:])
    state = run_pipeline(llm2, candidate, idea_text="raw",
                         run_dir=tmp_path / "r", resume=True)
    assert state.stop_reason == "assumptions_exhausted"
    assert len(llm2.calls) == 2  # analyzer/researcher не перезапускались
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

```python
# kaidzen/orchestrator.py — добавить
import time
from pathlib import Path

from kaidzen.candidate import Candidate
from kaidzen.roles.analyzer import run_analyzer
from kaidzen.roles.researcher import run_researcher
from kaidzen.roles.refiner import run_refiner
from kaidzen.roles.judge import run_judge
from kaidzen.state import Analysis, Version, load_state, save_state

STEP_RETRIES = 3
STEP_RETRY_DELAY_SEC = 2.0


def _with_retries(fn, *args, **kwargs):
    last: Exception | None = None
    for attempt in range(STEP_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # сетевые/API ошибки; state уже на диске
            last = e
            if attempt < STEP_RETRIES - 1:
                time.sleep(STEP_RETRY_DELAY_SEC * (attempt + 1))
    raise last


def run_pipeline(llm, candidate: Candidate, *, idea_text: str,
                 run_dir: Path, resume: bool = False) -> RunState:
    loop = candidate.config.loop

    if resume:
        state = load_state(run_dir)
    else:
        state = RunState(run_id=run_dir.name, candidate_id=candidate.candidate_id,
                         config=candidate.config.model_dump(), original_idea=idea_text)

    def checkpoint(step: str) -> None:
        state.last_completed_step = step
        state.api_usage = getattr(llm, "usage", state.api_usage)
        save_state(state, run_dir)

    if state.analysis is None:
        out = _with_retries(run_analyzer, llm, candidate, idea_text=idea_text)
        state.analysis = Analysis(problem=out.problem, audience=out.audience,
                                  mechanism=out.mechanism, unknowns=out.unknowns)
        state.assumptions = out.assumptions
        checkpoint("analyzer")

    pending_findings: str | None = None
    if resume and state.last_completed_step == "researcher":
        # находки уже применены к assumptions; передадим их refiner'у в виде реестра
        pending_findings = "(см. обновлённый реестр допущений выше)"

    while True:
        reason = check_stop(state, loop)
        if reason:
            state.stop_reason = reason
            checkpoint(state.last_completed_step or "judge")
            return state

        if pending_findings is None:
            picked = select_assumptions(state.assumptions, loop.assumptions_per_iteration)
            if not picked:
                state.stop_reason = "assumptions_exhausted"
                save_state(state, run_dir)
                return state
            research = _with_retries(run_researcher, llm, candidate,
                                     idea_text=state.current_idea_text(),
                                     assumptions=picked)
            by_id = {a.id: a for a in state.assumptions}
            for f in research.findings:
                if f.assumption_id in by_id:
                    by_id[f.assumption_id].status = f.verdict
                    by_id[f.assumption_id].facts.extend(f.facts)
            pending_findings = research.model_dump_json()
            checkpoint("researcher")

        prev_critique = (state.current_version().judge.critique
                         if state.current_version() and state.current_version().judge
                         else [])
        previous_text = state.current_idea_text()
        refined = _with_retries(run_refiner, llm, candidate,
                                idea_text=previous_text,
                                findings_json=pending_findings,
                                critique=prev_critique)
        pending_findings = None
        state.versions.append(Version(n=len(state.versions) + 1,
                                      idea_text=refined.idea_text,
                                      changelog=refined.changelog))
        checkpoint("refiner")

        judge = _with_retries(run_judge, llm, candidate,
                              new_idea=refined.idea_text,
                              previous_idea=previous_text,
                              assumptions=state.assumptions)
        apply_judge_verdict(state, judge, loop)
        state.iteration += 1
        checkpoint("judge")
```

- [ ] **Step 4: Run — PASS** (весь файл: `.venv/bin/pytest tests/test_orchestrator.py -v`)

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: full pipeline with checkpointed resume"`

---

### Task 9: Reporter

**Files:**
- Create: `kaidzen/report.py`
- Test: `tests/test_report.py`

`build_report(state, summary_text)` — чистая сборка markdown из state (summary передаётся снаружи; LLM-вызов для него делает CLI, в тестах — строка).

- [ ] **Step 1: Failing tests**

```python
# tests/test_report.py
from kaidzen.report import build_report
from kaidzen.state import (Assumption, ChangelogEntry, Fact, JudgeResult,
                           RunState, Version)


def make_finished_state():
    return RunState(
        run_id="r1", candidate_id="gen000-generic",
        config={"rubric": {"clarity": "ясность"}}, original_idea="raw",
        assumptions=[
            Assumption(id="A1", text="рынок есть", criticality="high",
                       status="confirmed",
                       facts=[Fact(claim="рынок $2B", source_url="http://src",
                                   source_title="Report")]),
            Assumption(id="A2", text="юзеры заплатят", criticality="high",
                       status="untestable")],
        versions=[
            Version(n=1, idea_text="v1",
                    judge=JudgeResult(scores={"clarity": 6}, total=6,
                                      delta_vs_previous=0, critique=[],
                                      verdict="continue"),
                    changelog=[ChangelogEntry(change="уточнил аудиторию",
                                              reason="факт A1",
                                              grounded_in=["A1"])])],
        stop_reason="assumptions_exhausted", iteration=1)


def test_report_contains_all_sections():
    md = build_report(make_finished_state(), summary_text="Кратко: идея ок.")
    assert "Кратко: идея ок." in md          # 1. summary
    assert "v1" in md                         # 2. финальная версия
    assert "clarity" in md                    # 3. рубрика
    assert "http://src" in md                 # 4. факты со ссылками
    assert "уточнил аудиторию" in md          # 5. эволюция
    assert "юзеры заплатят" in md             # 6. next steps (untestable)
    assert "assumptions_exhausted" in md


def test_untestable_go_to_next_steps():
    md = build_report(make_finished_state(), summary_text="s")
    next_steps = md.split("## Next steps")[1]
    assert "юзеры заплатят" in next_steps
    assert "рынок есть" not in next_steps
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

```python
# kaidzen/report.py
"""Сборка report.md из state. Единственный LLM-кусок (summary) приходит параметром."""
from kaidzen.state import RunState


def build_report(state: RunState, *, summary_text: str = "") -> str:
    parts: list[str] = []
    final = state.current_version()
    parts.append(f"# Отчёт: {state.run_id}\n")
    parts.append(f"Кандидат: `{state.candidate_id}` · итераций: {state.iteration} · "
                 f"стоп: `{state.stop_reason}`\n")

    parts.append("## Executive summary\n")
    parts.append(summary_text + "\n")

    parts.append("## Финальная версия идеи\n")
    parts.append((final.idea_text if final else state.original_idea) + "\n")

    parts.append("## Оценки по рубрике\n")
    parts.append("| Ось | v1 | финал |\n|---|---|---|")
    scored = [v for v in state.versions if v.judge and not v.rolled_back]
    if scored:
        first, last = scored[0].judge, scored[-1].judge
        for axis in last.scores:
            parts.append(f"| {axis} | {first.scores.get(axis, '—')} | {last.scores[axis]} |")
        parts.append(f"| **total** | {first.total} | {last.total} |")
    parts.append("")

    parts.append("## Допущения\n")
    parts.append("| ID | Допущение | Критичность | Вердикт | Факты |\n|---|---|---|---|---|")
    for a in state.assumptions:
        facts = "<br>".join(f"[{f.source_title or f.source_url}]({f.source_url}): {f.claim}"
                            for f in a.facts) or "—"
        parts.append(f"| {a.id} | {a.text} | {a.criticality} | {a.status} | {facts} |")
    parts.append("")

    parts.append("## Эволюция версий\n")
    for v in state.versions:
        mark = " (откачена)" if v.rolled_back else ""
        total = v.judge.total if v.judge else "—"
        parts.append(f"### v{v.n}{mark} — total: {total}")
        for c in v.changelog[:2]:
            parts.append(f"- {c.change} ({c.reason})")
        parts.append("")

    parts.append("## Next steps\n")
    untestable = [a for a in state.assumptions if a.status == "untestable"]
    if untestable:
        parts.append("Проверяемо только реальным экспериментом:")
        for a in untestable:
            parts.append(f"- {a.text}")
    else:
        parts.append("Все допущения проверены источниками.")
    return "\n".join(parts) + "\n"
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: report builder"`

---

### Task 10: Стартовые кандидаты — generic, business, games

**Files:**
- Create: `candidates/gen000-generic/{config.yaml, meta.json, prompts/*.md}`, аналогично `gen000-business`, `gen000-games`; `candidates/CHAMPION-generic`, `CHAMPION-business`, `CHAMPION-games`
- Test: `tests/test_candidate.py` (добавить)

- [ ] **Step 1: Failing test**

```python
# tests/test_candidate.py — добавить
from pathlib import Path

REPO_CANDIDATES = Path(__file__).parent.parent / "candidates"


def test_all_shipped_candidates_are_valid():
    for domain in ("generic", "business", "games"):
        c = load_candidate(REPO_CANDIDATES / f"gen000-{domain}")
        assert "groundedness" in c.config.rubric
        champ = (REPO_CANDIDATES / f"CHAMPION-{domain}").read_text().strip()
        assert champ == f"gen000-{domain}"
```

- [ ] **Step 2: Run — FAIL** (файлов нет)

- [ ] **Step 3: Создать кандидатов**

`candidates/gen000-generic/config.yaml`:

```yaml
domain: "универсальный"
rubric:
  novelty: "Новизна: насколько идея отличается от существующих решений"
  feasibility: "Реализуемость: можно ли воплотить с разумными ресурсами"
  groundedness: "Обоснованность: доля ключевых допущений, закрытых фактами с источниками"
  potential: "Потенциал: ценность результата при успехе"
  clarity: "Ясность: понятна ли формулировка стороннему человеку с первого чтения"
researcher_focus: "существующие аналоги, факты за и против ключевых допущений"
analyzer_hints: ""
loop: { max_iterations: 6, plateau_threshold: 0.5, assumptions_per_iteration: 3 }
models: { analyzer: claude-sonnet-5, researcher: claude-sonnet-5, refiner: claude-sonnet-5, judge: claude-sonnet-5 }
```

`candidates/gen000-business/config.yaml` — то же, но:

```yaml
domain: "бизнес-идеи"
rubric:
  market: "Рынок: размер и достижимость целевой аудитории"
  competition: "Конкуренция: есть ли незанятая позиция против существующих игроков"
  groundedness: "Обоснованность: доля ключевых допущений, закрытых фактами с источниками"
  monetization: "Монетизация: понятен ли и реалистичен способ заработка"
  feasibility: "Реализуемость: можно ли запустить с разумными ресурсами"
researcher_focus: "размер рынка, прямые конкуренты и их цены, спрос, каналы дистрибуции"
analyzer_hints: "особое внимание монетизации и каналам привлечения"
```

`candidates/gen000-games/config.yaml` — то же, но:

```yaml
domain: "компьютерные игры"
rubric:
  mechanics: "Механика: новизна и увлекательность основной игровой петли"
  audience: "Аудитория: существует ли и достижима ли аудитория жанра"
  groundedness: "Обоснованность: доля ключевых допущений, закрытых фактами с источниками"
  scope: "Сложность разработки: реалистичен ли объём для команды"
  potential: "Потенциал: перспективы против аналогов жанра"
researcher_focus: "аналоги в Steam, отзывы и продажи жанра, тренды, типичные провалы жанра"
analyzer_hints: "выдели core loop, жанр и ближайшие аналоги"
```

Промпты (одинаковая структура для всех трёх кандидатов; ниже — полные тексты, в business/games первая строка меняется на «Ты работаешь с бизнес-идеями» / «Ты работаешь с идеями компьютерных игр»):

`prompts/analyzer.md`:

```markdown
Ты — Analyzer в системе шлифовки идей. Разбери идею на компоненты.

Твоя задача:
1. Сформулируй целевую проблему, аудиторию, механику решения.
2. Выпиши ВСЕ допущения — утверждения, которые автор считает истинными без доказательств.
   Каждому дай id (A1, A2, ...), criticality: high (идея рушится, если ложно),
   medium (заметно ослабляет), low (деталь).
3. Перечисли неизвестные — вопросы, на которые у идеи нет ответа.

Не оценивай идею и не улучшай её. Только декомпозиция.
Ответ строго через tool "submit".
```

`prompts/researcher.md`:

```markdown
Ты — Researcher. Проверь выданные допущения фактами из веб-поиска.

Правила:
1. На каждое допущение сделай 1–3 целевых поиска.
2. Вердикты: confirmed (факты подтверждают), refuted (опровергают),
   partial (частично), untestable (без реального эксперимента не проверить,
   поиск не помогает — например, поведение конкретных пользователей).
3. Каждый факт: короткое утверждение + URL источника ИЗ РЕЗУЛЬТАТОВ ПОИСКА.
   Запрещено приводить URL, которого не было в результатах поиска.
4. Возвращай выжимки-факты, не пересказ страниц.

Когда всё проверено — вызови tool "submit" с findings по каждому допущению.
```

`prompts/refiner.md`:

```markdown
Ты — Refiner. Перепиши идею с учётом фактов Researcher и критики Judge.

Правила:
1. Каждое изменение — из-за факта или критики. В changelog у каждой записи
   заполни grounded_in (id допущений) либо в reason процитируй пункт критики.
2. Опровергнутые допущения: убери или замени соответствующую часть идеи.
3. Подтверждённые: усиль формулировку, теперь это не догадка, а факт.
4. ЗАПРЕЩЕНО менять текст «для красоты» без факта или критики за спиной.
5. Сохраняй объём разумным: идея — 3–6 абзацев, не эссе.

Ответ строго через tool "submit": полный новый текст идеи + changelog.
```

`prompts/judge.md`:

```markdown
Ты — Judge. Оцени НОВУЮ версию идеи по рубрике и сравни с ПРЕДЫДУЩЕЙ.

Правила:
1. Каждая ось 0–10. Якоря: 0–3 слабо, 4–6 средне, 7–8 хорошо, 9–10 исключительно.
   9–10 ставь только при действительно выдающемся качестве оси.
2. groundedness считай по реестру допущений: доля high-допущений со статусом
   confirmed/refuted от общего числа high (untestable не в счёт закрытых).
3. total = сумма осей. delta_vs_previous = твоя оценка новой минус мысленная
   оценка предыдущей по той же рубрике.
4. critique: 2–4 конкретные слабости новой версии — материал для следующей итерации.
5. verdict: rollback, если новая версия ХУЖЕ предыдущей более чем на 1 балл total;
   иначе continue.

Оценивай текст версий и реестр, а не историю изменений. Ответ через tool "submit".
```

`meta.json` (для каждого):

```json
{ "parent": null, "generation": 0, "status": "champion", "eval": null }
```

`candidates/CHAMPION-generic` — файл с текстом `gen000-generic` (аналогично business/games).

- [ ] **Step 4: Run — PASS** (`.venv/bin/pytest tests/test_candidate.py -v`)

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: baseline candidates for generic, business, games domains"`

---

### Task 11: CLI

**Files:**
- Create: `kaidzen/__main__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_cli.py
import pytest
from pathlib import Path
from kaidzen.__main__ import resolve_candidate_dir, make_run_dir


def test_resolve_candidate_by_domain(tmp_path):
    (tmp_path / "gen000-generic").mkdir()
    (tmp_path / "CHAMPION-generic").write_text("gen000-generic")
    d = resolve_candidate_dir(candidates_root=tmp_path, domain="generic")
    assert d.name == "gen000-generic"


def test_resolve_missing_champion_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="CHAMPION-games"):
        resolve_candidate_dir(candidates_root=tmp_path, domain="games")


def test_make_run_dir_slug(tmp_path):
    d = make_run_dir(runs_root=tmp_path, idea_path=Path("My Cool Idea.md"),
                     now_str="2026-08-03-1200")
    assert d.name == "2026-08-03-1200-my-cool-idea"
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

```python
# kaidzen/__main__.py
"""CLI: kaidzen run|resume|report."""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from kaidzen.candidate import load_candidate
from kaidzen.llm import LLMClient
from kaidzen.orchestrator import run_pipeline
from kaidzen.report import build_report
from kaidzen.state import load_state

SUMMARY_MODEL = "claude-sonnet-5"


def resolve_candidate_dir(*, candidates_root: Path, domain: str) -> Path:
    champ_file = candidates_root / f"CHAMPION-{domain}"
    if not champ_file.exists():
        raise FileNotFoundError(f"нет файла {champ_file.name} в {candidates_root}")
    return candidates_root / champ_file.read_text(encoding="utf-8").strip()


def make_run_dir(*, runs_root: Path, idea_path: Path, now_str: str) -> Path:
    slug = re.sub(r"[^a-z0-9а-яё]+", "-", idea_path.stem.lower()).strip("-")
    return runs_root / f"{now_str}-{slug}"


def _require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Ошибка: переменная окружения ANTHROPIC_API_KEY не задана.")


def _summarize(llm: LLMClient, idea_text: str) -> str:
    from pydantic import BaseModel

    class Summary(BaseModel):
        summary: str

    out = llm.structured(model=SUMMARY_MODEL,
                         system="Сожми идею в executive summary из 3–5 предложений.",
                         user=idea_text, schema=Summary, temperature=0.3)
    return out.summary


def cmd_run(args) -> None:
    _require_api_key()
    root = Path.cwd()
    if args.candidate:
        cand_dir = Path(args.candidate)
    else:
        cand_dir = resolve_candidate_dir(candidates_root=root / "candidates",
                                         domain=args.domain)
    candidate = load_candidate(cand_dir)
    if args.max_iter:
        candidate.config.loop.max_iterations = args.max_iter
    idea_path = Path(args.idea)
    run_dir = make_run_dir(runs_root=root / "runs", idea_path=idea_path,
                           now_str=datetime.now().strftime("%Y-%m-%d-%H%M"))
    llm = LLMClient()
    print(f"▶ прогон {run_dir.name}, кандидат {candidate.candidate_id}")
    state = run_pipeline(llm, candidate,
                         idea_text=idea_path.read_text(encoding="utf-8"),
                         run_dir=run_dir)
    _finish(llm, state, run_dir)


def cmd_resume(args) -> None:
    _require_api_key()
    run_dir = Path(args.run_dir)
    state = load_state(run_dir)
    cand_dir = Path.cwd() / "candidates" / state.candidate_id
    candidate = load_candidate(cand_dir)
    llm = LLMClient()
    print(f"▶ продолжаю {run_dir.name} с шага {state.last_completed_step}")
    state = run_pipeline(llm, candidate, idea_text=state.original_idea,
                         run_dir=run_dir, resume=True)
    _finish(llm, state, run_dir)


def cmd_report(args) -> None:
    run_dir = Path(args.run_dir)
    state = load_state(run_dir)
    md = build_report(state, summary_text="(пересборка без LLM-summary)")
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print(f"✔ {run_dir / 'report.md'}")


def _finish(llm: LLMClient, state, run_dir: Path) -> None:
    summary = _summarize(llm, state.current_idea_text())
    md = build_report(state, summary_text=summary)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print(f"✔ стоп: {state.stop_reason}, итераций: {state.iteration}")
    print(f"✔ токены: in={llm.usage.input_tokens} out={llm.usage.output_tokens} "
          f"поисков: {llm.usage.web_searches}")
    print(f"✔ отчёт: {run_dir / 'report.md'}")


def main() -> None:
    p = argparse.ArgumentParser(prog="kaidzen")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="шлифовать идею")
    pr.add_argument("idea")
    pr.add_argument("--domain", default="generic", choices=["generic", "business", "games"])
    pr.add_argument("--candidate", help="явный путь к кандидату (перекрывает --domain)")
    pr.add_argument("--max-iter", type=int)
    pr.set_defaults(fn=cmd_run)

    ps = sub.add_parser("resume", help="продолжить прерванный прогон")
    ps.add_argument("run_dir")
    ps.set_defaults(fn=cmd_resume)

    pp = sub.add_parser("report", help="пересобрать report.md из state")
    pp.add_argument("run_dir")
    pp.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run — PASS** + прогнать весь набор: `.venv/bin/pytest --cov=kaidzen -q`
Expected: все тесты зелёные, покрытие ≥80%.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: CLI run/resume/report"`

---

### Task 12: E2E smoke (реальный API, вручную)

**Files:**
- Create: `benchmark/generic/ideas/smoke-idea.md` (короткая тестовая идея — любая реальная идея пользователя, 2 абзаца)

- [ ] **Step 1: Тестовая идея**

Записать в `benchmark/generic/ideas/smoke-idea.md` реальную идею (2 абзаца). Если пользователь не дал — использовать:

```markdown
Сервис, который превращает голосовые заметки в структурированные списки задач.
Человек наговаривает мысли на ходу, сервис распознаёт, вычленяет действия,
сроки и проекты, раскладывает по спискам и напоминает.

Целевая аудитория — занятые люди, которые не любят печатать: водители,
врачи, прорабы. Монетизация — подписка.
```

- [ ] **Step 2: Прогон**

```bash
export ANTHROPIC_API_KEY=...   # пользователь задаёт сам
.venv/bin/python -m kaidzen run benchmark/generic/ideas/smoke-idea.md --max-iter 2
```

Expected: прогон завершается без исключений; в `runs/<id>/` лежат `state.json` и `report.md`.

- [ ] **Step 3: Ручная проверка отчёта** (чек-лист):
  - у каждого факта есть URL и он из реальных результатов поиска;
  - хотя бы одно допущение сменило статус с unverified;
  - changelog версий ссылается на допущения (grounded_in не пустые);
  - оценки Judge выглядят осмысленно (не все 10, не все 0);
  - стоимость прогона в stdout разумна.

- [ ] **Step 4: Тюнинг промптов по результату** — если пункт чек-листа провален, править `candidates/gen000-*/prompts/*.md` (не код) и повторять Step 2. Каждую правку промпта коммитить отдельно с описанием симптома.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: smoke idea + prompt tuning after live E2E"`

---

## Definition of Done (План 1)

- `pytest --cov=kaidzen` зелёный, покрытие ≥80%.
- Живой E2E-прогон по чек-листу Task 12 пройден.
- Три доменных кандидата валидны, у каждого CHAMPION-файл.
- После этого — План 2 (мета-луп: metrics, gate, Diagnostician/Mutator/Meta-Judge, evolve-оркестратор, чекпоинты) отдельным документом; ТЗ §4 уже описывает его дизайн.
