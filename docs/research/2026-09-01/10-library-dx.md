# Kaidzen: DX, API, packaging, docs, extensibility

Метки: **[strong]** проверено в коде репозитория или процитировано из первоисточника · **[suggestive]** практика с примерами · **[speculative]** гипотеза.

## Нулевая находка, обесценивающая всё остальное

**Установленный пакет сейчас не запускается нигде, кроме корня репозитория. [strong]**

- `kaidzen/__main__.py:50-53` — `CANDIDATES_ROOT = Path("candidates")`, `RUNS_ROOT = Path("runs")`, `BENCHMARK_ROOT`, `EVOLVE_ROOT` — все относительные к cwd.
- `pyproject.toml` кладёт в wheel только `kaidzen*` плюс `kaidzen/prompts/**/*.md`. Каталог `candidates/` лежит в корне репо и **в дистрибутив не попадает**.
- `pip install kaidzen && cd /tmp && python -m kaidzen run idea.md` падает на `FileNotFoundError: нет файла-указателя чемпиона: candidates/CHAMPION-generic`.
- `kaidzen/__init__.py` — **0 байт**. Публичного API нет вообще.
- `[project.scripts]` отсутствует → команды `kaidzen` нет, `uvx kaidzen` невозможен.

**Проект сегодня — не библиотека и не CLI-утилита, а клонируемый репозиторий.**

## (a) Публичный API

### Что показывает разведка сопоставимых библиотек

| Библиотека | Точка входа | Типизация результата | События | Инъекция бэкенда |
|---|---|---|---|---|
| DSPy | модуль-как-callable + `dspy.configure`/`context` | `Prediction` (attribute bag) | `StreamResponse` vs `StatusMessage`, диспетчеризация `isinstance` | строка `"provider/model"` **или** subclass `BaseLM.forward` |
| instructor | патч чужого клиента, `from_provider("openai/gpt-4o")` | pydantic (`response_model`) | `Partial[T]` / `create_iterable` | `Mode` как стратегия |
| LiteLLM | одна функция `completion()` | `ModelResponse` (чужая форма) | OpenAI-совместимые чанки | префикс провайдера в строке модели |
| inspect-ai | `@task`-фабрика + `eval()` → `list[EvalLog]` | dataclass-семейство `EvalLog` | пост-фактум чтение лога в 3 уровня детализации | `@modelapi` возвращает **класс**, не экземпляр, через entry points |
| smolagents | `agent.run(task, stream=...)` | dataclass `RunResult` | один union `StreamEvent` + `step_callbacks` | subclass `Model.generate` |
| GPT-Researcher | async-only, две фазы | нетипизировано | три параллельных канала, нетипизированные dict'ы | — |

Три вывода:
1. **Async posture.** Ни одна из шести не выбрала «async-ядро + sync-обёртка». Доминирует **«зеркальные близнецы»**: `completion`/`acompletion`, `call`/`acall`, `Instructor`/`AsyncInstructor` (отдельный класс, не обёртка) **[strong]**. У Kaidzen ядро синхронное, а `backends/claude_agent.py` уже держит синхронный мост через `anyio` — близнецы стоят дёшево.
2. **События должны быть типизированы union'ом.** Сейчас `on_step: Callable[[str, RunState], None]`, а предупреждения кодируются префиксом строки `"warning: <текст>"`, который `ProgressPrinter._print_warning` парсит обратно срезом **[strong]**. Антипаттерн GPT-Researcher.
3. **`@modelapi`-паттерн inspect-ai — лучшее решение опциональных зависимостей**: декоратор регистрирует **функцию, возвращающую класс**, «done so that you can separate the registration of models from the importing of libraries» **[strong]**. Kaidzen импортирует `anthropic`, `openai` и `claude_agent_sdk` безусловно в `registry.py` — все три в обязательных `dependencies`.

### Сквозное решение: `Workspace`
Решение проблемы cwd и дизайн API — одна и та же вещь.

```python
# kaidzen/__init__.py — сейчас пустой
from kaidzen.workspace import Workspace
from kaidzen.api import run, arun, RunResult, evolve
from kaidzen.events import Event, StepCompleted, Warning, RunFinished
from kaidzen.backends import LLMBackend, register_backend
__version__ = "0.2.0"
```

```python
class Workspace(BaseModel):
    root: Path
    candidates_root: Path
    runs_root: Path
    evolve_root: Path
    benchmark_root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "Workspace":
        """Ищет вверх по дереву каталог с candidates/ — как git ищет .git,
        а ruff/pytest ищут pyproject.toml. Не нашли — ошибка с подсказкой
        `kaidzen init`."""

    @classmethod
    def bundled(cls, runs_root: Path) -> "Workspace":
        """Кандидаты из самого пакета (importlib.resources), прогоны —
        куда попросили. Делает `uvx kaidzen run` рабочим без единого файла
        в текущем каталоге."""
```

Второй конструктор ключевой. Для него встроенные кандидаты переселить в `kaidzen/candidates/` и читать через `importlib.resources.files("kaidzen") / "candidates"`, а `candidates/` в корне репо оставить рабочим каталогом эволюции.

### Основная функция
```python
def run(
    idea: str | Path, *,
    domain: str = "generic",
    candidate: Path | Candidate | None = None,
    workspace: Workspace | None = None,
    max_iterations: int | None = None,
    backends: Mapping[str, LLMBackend] | None = None,   # подмена транспорта
    on_event: Callable[[Event], None] | None = None,
    dry_run: bool = False,
) -> RunResult: ...

async def arun(...) -> RunResult: ...          # близнец, не обёртка
def stream(...) -> Iterator[Event]: ...        # тот же прогон как генератор
```
- `idea: str | Path` — библиотечный пользователь чаще держит текст в памяти.
- `backends` опционален, по умолчанию из конфига кандидата. Сейчас вызывающий обязан сам сделать `build_backends()` + `backends_by_role()` — этим занимается приватный `__main__._build_role_backends` **[strong]**.
- `candidate: Path | Candidate` — приём загруженного `Candidate` даёт бесплатный custom prompt set.
- Разделение `run`/`stream` предпочтительнее флага, меняющего тип возврата (smolagents выбрал флаг и получил union-возврат) **[suggestive]**.

### Типизированный результат
`RunState` уже pydantic-модель. Нужна тонкая обёртка:
```python
class RunResult(BaseModel):
    state: RunState
    run_dir: Path
    report_path: Path | None
    @property
    def idea(self) -> str: ...
    @property
    def scores(self) -> dict[str, float]: ...
    @property
    def usage(self) -> ApiUsage: ...
    @property
    def metrics(self) -> RunMetrics: ...
    def report(self) -> str: ...
```
Это трёхуровневая схема чтения из inspect-ai в миниатюре **[suggestive]**.

### События
```python
class StepCompleted(Event):
    step: Literal["analyzer", "researcher", "refiner", "judge"]
    iteration: int
    state: RunState

class AssumptionsChecked(Event):
    verdicts: dict[str, AssumptionStatus]   # то, что ProgressPrinter сейчас
                                            # вычисляет сам, храня _last_statuses
class Scored(Event):
    total: float; delta: float; rolled_back: bool
class Warning(Event):
    message: str          # больше не префикс в строке шага
class RunFinished(Event):
    stop_reason: str; usage: ApiUsage
```
Выигрыш конкретный: `ProgressPrinter` **хранит `_last_statuses` и диффит статусы сам** (`__main__.py:200-262`) **[strong]**. Событие `AssumptionsChecked` переносит это знание в оркестратор, где оно и так есть. Тот же поток бесплатно превращается в `--json` (JSONL), MCP progress notifications и вывод GitHub Action.

Обратная совместимость не нужна — глобальные инструкции проекта предписывают удалять устаревшие пути. Заодно снести `kaidzen/llm.py`: это compat-шим (`LLMClient = AnthropicApiBackend`) в проекте без единого релиза **[strong]**.

## (b) Plugin-архитектура

### Что уже сделано правильно
`LLMBackend` (ABC) с одним абстрактным методом `structured()` и **классовым атрибутом способности** `supports_web_search: bool` — почти буквально `ModelAPI` из inspect-ai **[strong]**. `supports_web_search(backend_type)` в `registry.py` спрашивает способность у класса **до сборки и до единого платного вызова** — грамотнее, чем у большинства сравниваемых библиотек.

### Что закрыто

| Точка расширения | Сейчас | Проблема |
|---|---|---|
| Бэкенды | `BACKEND_CLASSES` — литеральный dict | третий бэкенд требует правки апстрима **[strong]** |
| Роли | `ROLES` кортеж + `_check_roles_complete` бракует `unexpected` | своя роль невозможна **[strong]** |
| Метрики | `run_metrics()` — одна функция, `RunMetrics` фиксирована | своя ось Gate невозможна **[strong]** |
| Домены | `DOMAINS` как argparse `choices` | сторонний домен только через `--candidate <путь>` **[strong]** |
| Формат кандидата | `CandidateConfig` с `extra="forbid"`, **без поля версии** | любой новый ключ ломает старые кандидаты и наоборот **[strong]** |

### Entry points группы `kaidzen.backends`, ленивая регистрация
```toml
# в пакете плагина, например kaidzen-gemini
[project.entry-points."kaidzen.backends"]
gemini = "kaidzen_gemini:register"
```
```python
@lru_cache(maxsize=1)
def backend_classes() -> dict[str, type[LLMBackend]]:
    """Регистрируем ФУНКЦИЮ, возвращающую класс, а не сам класс.
    SDK провайдера импортируется только когда бэкенд нужен."""
    found = dict(BUILTIN)
    for ep in entry_points(group="kaidzen.backends"):
        found[ep.name] = ep.load()      # ep.load() -> callable -> class
    return found
```
Почему функция, а не класс: inspect-ai документирует выбор явно **[strong]**. Решает и вопрос extras.

**Python 3.12 убрал dict-интерфейс `entry_points()` [strong].** Kaidzen требует `>=3.12` → `entry_points(group=...)` без шимов. Ключевое: возвращает **только метаданные**, модуль плагина не импортируется до `.load()` — перечисление бэкендов в `--help` не тянет ни один SDK.

Из pytest взять **аварийные выключатели**: `KAIDZEN_DISABLE_PLUGIN_AUTOLOAD=1` и `-p no:name` **[strong]** — для инструмента, тратящего деньги пользователя, дешёвая предохранительная мера. Из Datasette — **иерархию конфигурации плагина** (кандидат → workspace → глобально) **[strong]**. Из `llm` Саймона Уиллисона — **паттерн `register(...)`-колбэка вместо возврата словаря** **[strong]**: хост владеет реестром, контролирует коллизии имён и привязывает происхождение. Там же дисциплина: *«you should never ship a plugin hook without releasing at least one plugin that uses it»* — точку расширения бэкендов надо валидировать, вынеся один из трёх существующих в отдельный пакет.

**Автозагрузка должна глушиться под тестами [strong].** Datasette: `if not hasattr(sys, "_called_from_test") and DATASETTE_LOAD_PLUGINS is None: pm.load_setuptools_entrypoints(...)`. При 524 герметичных тестах за 3.3 с посторонний плагин не должен уметь изменить результат.

### Роли и метрики: НЕ делать плагинами
Роль в Kaidzen — не просто промпт: у Analyzer своя pydantic-схема, свой шаг в `run_pipeline`, своё место в протоколе возобновления (`MID_ITERATION_STEPS`, `last_completed_step`). Сделать роли плагинами — вынести в плагин управление циклом **[speculative]**.

- **Роли** — оставить пять. Расширяемость уже есть и называется «кандидат».
- **Метрики** — плагин уместен и дёшев, метрика чистая: `RunState -> float`.
```python
@runtime_checkable
class Metric(Protocol):
    name: str
    higher_is_better: bool
    def __call__(self, state: RunState) -> float: ...
```
`Protocol` там, где важна только форма; ABC там, где есть что наследовать (`self.usage`, `_guard_search_performed`) **[suggestive]**.

**Оговорка [strong]:** `runtime_checkable` Protocol проверяет **только наличие имён**, не типы и не сигнатуры. `isinstance(obj, Metric)` пропустит объект, у которого `name` — это `int`. Protocol для **типа**, отдельная явная функция-валидатор для **рантайм-ворот**. Плюс: изменяемый атрибут в Protocol инвариантен — лечится read-only property.

### Версионирование формата кандидата
`CandidateConfig` имеет `extra="forbid"` и никакого поля версии. Любой будущий ключ ломает старые кандидаты **и** старый Kaidzen ломается на новых. Для системы, где кандидаты живут годами и логируются в `EVOLUTION-<domain>.json`, это мина.

```yaml
kaidzen_format: 1        # первое поле; читается ДО model_validate
domain: бизнес-идеи
```
```python
CURRENT_FORMAT = 1
SUPPORTED_FORMATS = (1,)

def load_candidate(path: Path) -> Candidate:
    data = yaml.safe_load(...)
    fmt = data.pop("kaidzen_format", 1)
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"{path} написан в формате кандидата v{fmt}, эта версия kaidzen "
            f"понимает {SUPPORTED_FORMATS}. Обновите kaidzen или запустите "
            f"`kaidzen candidate migrate {path}`.")
    data = MIGRATIONS[fmt](data)
    return Candidate(..., config=CandidateConfig.model_validate(data))
```

**Важная поправка [strong].** Из четырёх «version field» кейсов два **регрессировали**: Docker Compose объявил своё поле obsolete, dbt сделал опциональным с 1.5. Единственный работающий — Kubernetes, и работает не из-за поля, а из-за server-side конверсии и письменного deprecation-контракта. **Поле версии без машинерии миграций вырождается в мёртвый груз или магическую константу.** Поле — дешёвая часть; продукт — цепочка чистых функций `dict -> dict` и golden-file тесты по одному на историческую версию. **Писать `kaidzen_format:` имеет смысл только вместе с `MIGRATIONS` и фикстурами.**

Плюс: ошибка «формат новее» должна говорить **«обновите kaidzen»**, а не «невалидный конфиг». Аддитивные изменения версию **не** поднимают — иначе цепочка миграций превращается в шум (контракт «argument pruning» из pytest и Datasette).

**Sphinx разделяет `version` и `env_version` [strong]:** первое — человеческая версия расширения, второе — целое, версионирующее **форму сохранённых данных**, «must increment when stored data type, structure, or meaning changes». Без `env_version` расширение не имеет права хранить состояние. Следствие: если появится кеш разобранных кандидатов, версия формата обязана входить в ключ кеша.

**Discriminated union конфликтует с рантайм-обнаружением плагинов [strong].** `Field(discriminator=...)` требует статически известный `Union`. `backends: dict[str, dict]` сейчас не типизирован внутри, а discriminated union по `type` дал бы ошибки вида «Input tag 'openai_compt' does not match any of the expected tags». Для трёх встроенных бэкендов разумнее статический union сейчас и двухфазная валидация — когда появятся плагины **[speculative]**.

**Поле версии вводить до первого релиза на PyPI**, пока чужих кандидатов не существует.

## (c) CLI, по убыванию отдачи

**1. `[project.scripts]` + рабочая установка вне репозитория.**
```toml
[project.scripts]
kaidzen = "kaidzen.cli:main"
```
Включает `uvx kaidzen` — uv требует исполняемого скрипта **[strong]**. Имя пакета и команды совпадают → `uvx kaidzen run idea.md` заработает без флагов.

**2. `kaidzen init` + поиск workspace вверх по дереву.** Пара к `Workspace.discover()`.

**3. Осмысленные коды возврата.** Сейчас `main()` ловит `FileNotFoundError | ValueError | BackendError` и делает `sys.exit(str(e))` — всё превращается в код 1 **[strong]**. Есть готовые прецеденты: **rclone кодирует retryable vs fatal прямо в коде** (`5` — «Temporary error (one that more retries might fix)», `7` — «Fatal error»); **restic** добавляет `3` (частичный успех) и `130` (отмена сигналом) плюс правило прямой совместимости **[strong]**.

| Код | Значение | Прецедент |
|---|---|---|
| 0 | успех | — |
| 1 | общая ошибка | — |
| 2 | ошибка использования / конфига | argparse |
| 3 | частичный успех | restic |
| 5 | транзиентный сбой — retry поможет | rclone |
| 7 | фатальный — retry не поможет | rclone |
| 130 | прервано Ctrl+C после сохранения состояния | restic |

`RETRYABLE_ERRORS` уже разделяет транзиентные и фатальные — просто это не доходит до кода возврата. Код 130 важен: `run_pipeline` сохраняет состояние после каждого шага, и обёртка должна отличать «можно `resume`» от «сломалось».

**4. Дисциплина stdout/stderr.** clig.dev: «Send output to `stdout` … Anything that is machine readable should also go to `stdout`»; «Send messaging to `stderr`. Log messages, errors, and so on» **[strong]**. Сейчас весь прогресс идёт в stdout через `print()`. При `kaidzen report … | pbcopy` в буфер попадёт прогресс вперемешку с отчётом.

**5. `--format human|json|agent|quiet` (а не просто `--json`).** CLI `hf` от Hugging Face резолвит режим один раз на старте и, главное **[strong]**:
```python
if mode != OutputFormat.human:
    disable_progress_bars()
```
Прогресс — свойство человеческого режима, а не отдельный флаг. Детекция агента (`AI_AGENT`/`AGENT` env) — best-effort с правилом «detection must never make a process fail». Режим `quiet` печатает только id. **Для Kaidzen режим `agent` ценнее обычного `--json`: дефолтный бэкенд — `claude_agent_sdk`, инструмент по построению часто вызывается из-под агента.**

**6. `--dry-run` + оценка стоимости.** clig.dev: «Confirm before doing anything dangerous» **[strong]**. Многоминутный платный прогон — этот случай.
```python
count = client.messages.count_tokens(model=..., system=prompt, messages=[...])
# openai_compat: litellm.token_counter / litellm.cost_per_token
```
Три оговорки **[strong]**: возвращает **только `input_tokens`** (выходные до вызова неизвестны в принципе → честная оценка обязана быть вилкой); «Token counting provides an estimate without using caching logic» → **верхняя граница**, если кеширование в игре, и это надо писать в выводе; токенизатор сменился — на моделях 4.7+ тот же текст даёт **примерно на 30% больше токенов**, замеры со старых моделей не переиспользовать. Паттерн aider: **подсчёт токенов никогда не бросает исключение**, при сбое возвращает 0 с предупреждением.

**7. Прогресс: три режима, а не TTY-гард.** Изначальный совет «прятать прогресс, когда не TTY» — **неправильный**. Правильнее **прореживать**: restic эмитит статус ~10 раз/сек «regardless of whether the output is a terminal or a pipe», `RESTIC_PROGRESS_FPS` позволяет одно сообщение в минуту — «useful when capturing logs» **[strong]**. pip идёт тремя режимами: `on` (Rich), `off`, **`raw`** — машиночитаемые строки в stdout с rate-limit 4/сек:
```python
def write_progress(current: int, total: int) -> None:
    sys.stdout.write(f"Progress {current} of {total}\n")
    sys.stdout.flush()
```
**Для Kaidzen критично: прогон идёт минуты, и в CI молчание восемь минут неотличимо от зависания.** Текущий `sys.stdout.reconfigure(line_buffering=True)` — половина решения. (pip также прячет бар при создании, `visible=False`, «avoiding very short flashes».)

**8. UX возобновления.** Механика уже сильная: `write_atomically` с fsync файла, `os.replace` и fsync каталога — лучше, чем у большинства инструментов класса **[strong]**. Не хватает витрины: `kaidzen resume <dir>` должен печатать «продолжаю с шага refiner, итерация 3 из 6, потрачено 412K токенов».

**9. Не мигрировать на Typer.** Typer 0.27.2 тянет `shellingham`, `rich`, `annotated-doc`, `colorama` (вендоринг Click подтверждён — `click` из его `requires_dist` исчез). Часть претензий протухла: `typing.Literal` поддержан с 0.19.0 **[strong]**. Но остаются: нет API-reference документации, нет mutually-exclusive опций, нет positional-or-keyword. **Последнее существенно: `--approve/--reject` реализованы через `add_mutually_exclusive_group` — Typer этого не умеет вовсе.**

**Поправка про Click [strong]:** `click` 8.5.0 имеет `requires_dist: None` — **ноль рантайм-зависимостей**. Аргумент про вес для него не работает. Уточнение: argparse остаётся разумным дефолтом, но если понадобится плагинная система с subcommand'ами от третьих лиц — Click оправдан (`llm`, `datasette`, `sqlite-utils`, `black` все на нём). Cyclopts не легковесен: `attrs`, `rich`, `rich-rst`, `docstring-parser`.

## (d) Packaging и релиз

**Сборка.** Туториал PyPA использует **Hatchling по умолчанию** **[strong]**. Предметное преимущество: `force-include` затаскивает в wheel файлы откуда угодно — решает вопрос встроенных кандидатов без переноса каталога. Не блокер: setuptools с `package-data` тоже справится, если кандидаты переедут под `kaidzen/`.

**Extras по бэкендам.** Сейчас все три SDK обязательны, хотя дефолтный сценарий требует только `claude-agent-sdk`.
```toml
dependencies = ["anyio>=4.0", "pydantic>=2.7", "pyyaml>=6.0"]

[project.optional-dependencies]
subscription = ["claude-agent-sdk>=0.2.128"]
anthropic    = ["anthropic>=0.120"]
openai       = ["openai>=1.40"]
all          = ["kaidzen[subscription,anthropic,openai]"]

[dependency-groups]           # PEP 735 — НЕ публикуется в метаданных
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff", "mypy"]
```
PEP 735 разводит явно: extras — «user-facing optional features»; dependency-groups — «development tools, testing frameworks», в дистрибутив не попадают **[strong]**. Текущий `[project.optional-dependencies] dev` — ровно тот случай, ради которого PEP 735 вводился.

**Тонкость, которую нельзя проглядеть:** `registry.py` импортирует все три бэкенда на верхнем уровне. С extras это станет `ImportError`. **Ленивая регистрация — не украшение, а предусловие для extras.**

**Trusted Publishing.**
```yaml
jobs:
  pypi-publish:
    environment: pypi          # «optional, but strongly encouraged»
    permissions:
      id-token: write          # «IMPORTANT: this permission is mandatory»
    steps:
      - uses: pypa/gh-action-pypi-publish@release/v1
```
`id-token: write` обязателен, давать на уровне job **[strong]**. Токены генерируются под конкретный запуск и истекают с ним.

**Тестирование нижних границ.** uv: «When publishing libraries, it is recommended to separately run tests with `--resolution lowest` or `--resolution lowest-direct` in CI» **[strong]**. В `pyproject.toml` уже есть содержательный комментарий, почему нижняя граница `anthropic` — 0.120 («заявленная граница обещала установку, которая не смогла бы сделать ни вызова»). **Такую границу надо не комментировать, а проверять в CI** матрицей: latest и `--resolution lowest-direct`.

**Версионирование, когда продукт — промпты.** Отраслевой ответ есть:
- spaCy: модели — **отдельные версионируемые пакеты**, каждый декларирует требуемую версию spaCy **[strong]**.
- HF Hub: контент — git-репозитории, версии = ревизии/теги, потребитель пинит `revision` **[strong]**.
- HACS: версия контента = тег последнего релиза, иначе первые 7 символов коммита **[strong]**.

**Три независимые оси:**

| Ось | Где живёт | Правило |
|---|---|---|
| Версия библиотеки | `pyproject.toml` | SemVer, `0.x` пока API не устоялся |
| Версия формата кандидата | `kaidzen_format:` в `config.yaml` | целое, растёт только на несовместимом изменении |
| Версия набора промптов | `meta.json` кандидата | уже есть `generation`/`parent` — не хватает `requires_kaidzen` |

**Изменение промпта — не патч и не минор, а новый кандидат.** У Kaidzen это уже так (`gen001-a`, `gen002-a-161653`), и эволюционный лог — уже журнал версий. Не натягивать SemVer на промпты; признать, что кандидаты — отдельно версионируемые артефакты в смысле spaCy-моделей **[suggestive]**.

**Мелочи:** нет ни одного git-тега; нет CHANGELOG; `version = "0.1.0"` живёт только в pyproject (выставить `kaidzen.__version__` и брать одно из другого).

## (e) Документация и билингвальность

### README
**Art of README настаивает: usage идёт ПЕРЕД installation** — рекомендуемый порядок Name, One-liner, Usage, API, Installation, License, и «The ordering of the above was not chosen at random» **[strong]**. Механизм — cognitive funneling: читатель решает «моё / не моё» до того, как узнает, как ставить. Текущий README ставит Quick start (установка) выше содержательного.

**Демо-ассет.** Прогон идёт **минуты** — живую запись показывать нельзя. Инструмент — **VHS** от Charmbracelet: сценарий-`.tape` с `Type`/`Enter`/`Sleep`/`Output`, вывод в GIF/MP4/WebM, официальный `charmbracelet/vhs-action` для обновления в CI **[strong]**. Скриптованный терминал позволяет показать сжатую версию прогона честно — это монтаж, не подделка.

**Реальный вывод.** У Kaidzen редкий актив: `runs/` и `evolve/` с настоящими отчётами. Один сквозной пример с реальной таблицей допущений, вердиктами и ссылками стоит больше, чем весь остальной README. Аналогично история про то, как эволюция сама попыталась смухлевать, — это лучший абзац README, потому что доказывает, что систему реально гоняли.

### Билингвальность — главный вопрос
Диагноз точнее, чем «README английский, промпты русские». **Английский README обещает английский инструмент, а первая же команда отвечает по-русски:** `_print_start` печатает «Прогон:», «Кандидат:», все сообщения об ошибках русские (`«не задан 'api_key_env'»`, `«рубрика должна содержать ровно 5 осей»`), `domain: бизнес-идеи` в конфиге **[strong]**. **Разрыв не в документации, а между документацией и рантаймом.**

Прецедент однозначен. ClickHouse — проект российского происхождения, ставший международным — фиксирует правилом: «Comments should be written in English only», «All names must be in English», «Log messages must be written in English» **[strong]**. Переход прошёл через жёсткое разделение: **код, комментарии, логи и ошибки — английские; документация — переводимая; данные — какие есть.**

Про промпты: единственный аргумент за русские — качество модели на русском домене; он **не проверен** (ни бенчмарка на переведённых промптах, ни следа сравнения в `docs/specs/`) **[speculative]**. Это идеальная задача для самого Kaidzen: домен `business` с английскими промптами против русских — эволюционный эксперимент, который система умеет проводить. Пока он не проведён, держать промпты русскими надо по честной причине: «на них система разработана и отвалидирована».

**Три яруса, не два:**

| Слой | Язык | Почему |
|---|---|---|
| Код, докстринги, **сообщения об ошибках, вывод CLI** | English | правило ClickHouse; это интерфейс инструмента, а не контент **[strong]** |
| README, docs-сайт | English-first, русский вторым | вход в проект |
| Промпты, спеки, бенчмарк-идеи | Russian, с явной пометкой | данные и исследовательская история, не интерфейс |

**Самый дешёвый и недооценённый шаг — перевести сообщения об ошибках и вывод CLI, оставив внутренние комментарии как есть.** Комментарии в Kaidzen исключительно хороши (объясняют «почему»: почему fsync каталога, почему `SearchNotPerformed` наследуется от `ValueError`, почему нижняя граница anthropic — 0.120). Переводить их — потерять качество ради галочки; они не видны пользователю. Ошибки и вывод — видны.

Спеки **не переводить, а обрамлять**: одно английское оглавление с 3–5-строчными аннотациями. Дешевле перевода на порядок, и честнее.

### Docs-сайт
mkdocs-material поддерживает **одну каноническую язык на проект** («HTML5 only allows to set a single language per document»); мультиязычность — либо подпапка на язык плюс `extra.alternate`, либо `mkdocs-static-i18n` **[strong]**. Docusaurus i18n богаче, но тянет Node в Python-проект.

**На старте не поднимать сайт вообще.** Хороший README + `docs/` в репозитории закрывают потребность проекта без релиза на PyPI **[suggestive]**. Прямое следствие принципа «расти слоями».

## (f) Каналы дистрибуции

**1. Библиотечный импорт + `uvx` — наибольшая отдача, наименьшее усилие.** Следствие (a)+(d). `uvx kaidzen run idea.md` — самый короткий путь от твита до запуска **[strong]**.

**2. Claude Code plugin — дёшево и очень уместно.** Дефолтный бэкенд — `claude_agent_sdk`, то есть **аудитория Kaidzen и аудитория Claude Code — одни и те же люди с уже установленным `claude` CLI**. Плагин — каталог с `.claude-plugin/plugin.json` плюс `skills/<name>/SKILL.md`; тестируется через `claude --plugin-dir ./my-plugin`; распространяется через marketplace-репозиторий; есть публичный `claude-community` marketplace и `claude plugin validate` **[strong]**. Объём — часы.

**3. MCP-сервер — средняя отдача, реальный подводный камень.**
```python
@mcp.tool()
def polish_idea(idea: str, domain: str = "business") -> str:
    """Отполировать сырую идею фактами из веб-поиска."""
    return kaidzen.run(idea, domain=domain).report()
```
Тип-хинты и есть схема **[strong]**. Но многоминутная операция как MCP-tool — риск таймаута. Спека даёт один механизм: `progressToken` в `_meta` и `notifications/progress` с `progress`/`total`/`message`, причём «`progress` value MUST increase» и обе стороны «SHOULD implement rate limiting» **[strong]**. Типизированные события ложатся один-в-один. Честный вариант: два инструмента — `kaidzen_start` (возвращает `run_id` сразу) и `kaidzen_status(run_id)` **[suggestive]**.

**4. GitHub Action — низкая отдача.** Composite action пишется быстро **[strong]**, но сценарий «полировать идеи в CI» надуман: нет триггера в жизненном цикле репозитория, а платный многоминутный прогон в Action сомнителен.

## (g) Сводные рекомендации

| # | Рекомендация | Impact | Effort | Risk |
|---|---|---|---|---|
| 1 | `Workspace` + встроенные кандидаты в пакете + `[project.scripts]` | **Критический** | Средний | Низкий |
| 2 | Поле `kaidzen_format` + **цепочка миграций и golden-file тесты** | Высокий | Низкий | Низкий — **сделать до PyPI** |
| 3 | CI на GitHub Actions (сейчас `.github/` нет вовсе) | Высокий | Низкий | Низкий |
| 4 | Публичный API: `kaidzen.run/arun/stream` + `RunResult` | Высокий | Средний | Низкий |
| 5 | Типизированные события вместо `on_step(str, RunState)` | Высокий | Средний | Средний |
| 6 | Перевести ошибки и вывод CLI на английский (комментарии не трогать) | Высокий | Низкий | Низкий |
| 7 | Extras по бэкендам + ленивая регистрация | Средний | Средний | Средний |
| 8 | README: usage перед install + VHS-демо + реальный вывод | Средний | Низкий | Низкий |
| 9 | **Коды возврата (rclone/restic) + stdout/stderr дисциплина** | **Высокий** | Низкий | Низкий |
| 10 | `--dry-run` с оценкой стоимости (`count_tokens`, вилка, не бросает) | Средний | Средний | Низкий |
| 11 | Trusted Publishing + CI на `--resolution lowest-direct` | Средний | Низкий | Низкий |
| 12 | Entry points `kaidzen.backends` + `KAIDZEN_DISABLE_PLUGIN_AUTOLOAD` + глушение под тестами | Средний | Средний | Средний |
| 13 | `--format human\|json\|agent\|quiet` | Средний | Низкий | Низкий |
| 14 | Claude Code plugin | Средний | Низкий | Низкий |
| 15 | CONTRIBUTING + good-first-issue + pre-commit + ruff | Средний | Низкий | Низкий |
| 16 | **Три режима прогресса по образцу pip (`on`/`off`/`raw`) с прореживанием в non-TTY** | **Средний** | Средний | Средний |
| 17 | Метрики как `Protocol`-плагин (+ отдельный валидатор) | Низкий | Низкий | Низкий |
| 18 | MCP-сервер (start/status, не блокирующий tool) | Низкий | Средний | Средний |
| 19 | Снести `kaidzen/llm.py` (compat-шим до первого релиза) | Низкий | Низкий | Низкий |
| 20 | Docs-сайт (mkdocs-material) | Низкий | Средний | Низкий |
| 21 | GitHub Action | Очень низкий | Низкий | Низкий |
| 22 | **Миграция на Typer/Click** | **Отрицательный** | Средний | Средний |

### Механика контрибуции — данные, а не ритуалы
Mozilla (через opensource.guide): «contributors who received code reviews within 48 hours had a much higher rate of return and repeat contribution» **[strong]** — **скорость ревью бьёт качество CONTRIBUTING.md**. GitHub Open Source Survey 2017: «incomplete or confusing documentation is the biggest problem for open source users» **[strong]**. Метки good-first-issue работают через платформу **[strong]**.

Сильные стороны Kaidzen: 524 теста за 3.3 с без сети **[strong]**, комментарии «почему», модули разумного размера. Слабые: нет CI, CONTRIBUTING, pre-commit, ruff/mypy, тегов. Порядок: CI → ruff (заменяет flake8+black+isort+pydocstyle+pyupgrade+autoflake) → CONTRIBUTING → метки.

### Как принимать чужих кандидатов
Три модели: в репозитории (просто, но апстрим отвечает за чужие промпты); через entry points на PyPI (как pytest, который прямо предупреждает, что это «not a curated collection» **[strong]**); реестр-репозиторий с манифестом (HACS; Claude Code marketplace идёт дальше — одобренные плагины пинятся к commit SHA **[strong]**).

Для Kaidzen существенно то, чего нет у pytest-плагинов: **кандидат — это промпты, то есть текст, который пойдёт в модель от имени пользователя и потратит его деньги.** Prompt injection здесь не теоретический риск: злонамеренный `researcher.md` может заставить модель фабриковать источники — ровно то, против чего построена вся защита `SearchNotPerformed`/`groundedness` **[speculative, но механизм очевиден]**.

**Не открывать реестр рано.** Начать с awesome-list в README (ссылки на чужие репозитории, никакого доверия по умолчанию, установка через явный `--candidate <путь>`), и дополнить тем, чего нет ни у одного из рассмотренных реестров, но что у Kaidzen построено: **прогон кандидата по бенчмарку и метрикам Gate как условие включения в список.** У проекта, умеющего объективно измерять качество набора промптов, критерий отбора должен быть измеряемым, а не редакторским **[suggestive]** — пожалуй, сильнейшая дифференциация среди всех реестров.

## (h) Источники
**API:** DSPy https://dspy.ai/api/models/LM/ · instructor https://python.useinstructor.com/concepts/hooks/ · LiteLLM https://docs.litellm.ai/docs/completion/token_usage · inspect-ai https://inspect.aisi.org.uk/extensions-model-api.html · smolagents https://huggingface.co/docs/smolagents/en/reference/agents · GPT-Researcher https://docs.gptr.dev/docs/gpt-researcher/gptr/pip-package
**Плагины:** https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/ · https://docs.python.org/3/library/importlib.metadata.html · https://docs.pytest.org/en/stable/how-to/writing_plugins.html · https://docs.datasette.io/en/stable/writing_plugins.html · https://github.com/simonw/datasette/blob/main/datasette/plugins.py · https://llm.datasette.io/en/stable/plugins/plugin-hooks.html · https://peps.python.org/pep-0544/ · https://pydantic.dev/docs/validation/latest/concepts/unions/ · https://docs.docker.com/reference/compose-file/version-and-name/ · https://docs.getdbt.com/reference/project-configs/version · https://www.sphinx-doc.org/en/master/extdev/index.html
**Packaging:** https://packaging.python.org/en/latest/tutorials/packaging-projects/ · https://peps.python.org/pep-0735/ · https://docs.pypi.org/trusted-publishers/using-a-publisher/ · https://docs.astral.sh/uv/guides/tools/ · https://docs.astral.sh/uv/concepts/resolution/ · https://hatch.pypa.io/latest/config/build/ · https://spacy.io/usage/models · https://huggingface.co/docs/hub/repositories-getting-started
**CLI:** https://clig.dev/ · https://restic.readthedocs.io/en/stable/075_scripting.html · https://rclone.org/docs/ · https://github.com/pypa/pip/blob/main/src/pip/_internal/cli/progress_bars.py · https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/cli/_output.py · https://platform.claude.com/docs/en/build-with-claude/token-counting · https://cyclopts.readthedocs.io/en/latest/vs_typer/README.html · https://github.com/Aider-AI/aider/blob/main/aider/args.py
**Docs:** https://github.com/hackergrrl/art-of-readme · https://clickhouse.com/docs/development/style · https://squidfunk.github.io/mkdocs-material/setup/changing-the-language/ · https://github.com/charmbracelet/vhs
**Дистрибуция:** https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/progress · https://code.claude.com/docs/en/plugins · https://opensource.guide/building-community/ · https://www.hacs.xyz/docs/publish/start/
