# Kaidzen: инженерное качество, надёжность, стоимость, наблюдаемость

Метки: **[strong]** первичная документация/спецификация/код · **[suggestive]** вторичные источники, отраслевая практика · **[speculative]** экстраполяция.

## 1. Structured output

**Главная находка, экономящая недели: constrained decoding недоступен. [strong]**
Outlines, XGrammar, llguidance работают через маску логитов на каждом шаге декодирования — нужен локальный инференс-движок (vLLM, SGLang, TensorRT-LLM, transformers). Все четыре бэкенда Kaidzen — закрытые API или CLI поверх закрытого API. Подключить их к `claude_agent_sdk` или `api.deepseek.com` нельзя в принципе. **Вся ветка закрыта, не тратить время.**

Остаётся трёхуровневая лестница:

| Уровень | Гарантия | Где у Kaidzen |
|---|---|---|
| Server-side constrained (`strict: true` json_schema) | Схема гарантирована сервером | `openai_compat` в `MODE_JSON_SCHEMA` — корректно |
| Tool-use со схемой + `tool_choice` | Валидный JSON почти всегда, схема — нет | `anthropic_api` — корректно |
| «Верни JSON» в промпте + парсинг | Ничего | `claude_agent`, DeepSeek — здесь живут баги |

### Почему `json_extract.py` ломается [strong]
`_slice_outer_braces` берёт `text.find("{")` и `text.rfind("}")`. Четыре класса отказа:
1. **Два объекта в ответе.** Срез от первой `{` до последней `}` захватывает оба → `JSONDecodeError`. Хуже: при запятой между ними может получиться синтаксически валидный, но семантически чужой объект.
2. **Обрезанный по `max_tokens` ответ.** `rfind("}")` находит закрывающую скобку *вложенного* объекта → парсится частичный объект. Если недостающие поля имеют дефолты (`facts: list = Field(default_factory=list)` — именно так), Pydantic **проходит валидацию**, и данные молча теряются. В `anthropic_api` есть `ResponseTruncated` по `stop_reason`, в `claude_agent` и `openai_compat` — **нет**. Тихий отказ.
3. **`strict=False`** маскирует настоящую порчу.
4. **`_pick_json_text`** берёт последний блок текста, содержащий `{`. Комментарий модели после JSON со скобкой — возьмётся комментарий.

### Чем заменить [suggestive]
- `json-repair` (PyPI, чистый Python, без зависимостей) — чинит незакрытые скобки, оборванные строки, markdown-ограду, одинарные кавычки, trailing commas. **Детерминированно достраивает** обрезанный объект вместо отрезания хвоста наугад.
- `jiter` (Rust-парсер от команды Pydantic, уже стоит транзитивно через `openai`) — partial-режим для стриминга.
- **Instructor** — не брать целиком: не поддерживает Claude Agent SDK, а логика уже написана. Скопировать две идеи: класть текст ошибки валидации **отдельным сообщением в диалог** (уже так) и **логировать сырой ответ, не прошедший валидацию** (сейчас теряется).

### Структурный дефект важнее выбора парсера [strong]
Цикл retry-with-error-feedback **продублирован трижды**: `claude_agent.py:60-75`, `anthropic_api.py:80-125`, `openai_compat.py:66-76`. Три разных протокола сообщений, три набора ловимых исключений, одна общая константа `MAX_SCHEMA_RETRIES = 1`. Протокол чинили только в Anthropic-версии (коммит `2745fa3`). **Ровно тот класс, где живёт «сломанный протокол ретрая»: исправление в одном месте не доезжает до двух других.** Вынести цикл в `LLMBackend` как template method с двумя хуками: `_one_call(messages) -> raw` и `_feedback(raw, error) -> messages_delta`.

## 2. Тестирование без сети

Текущая схема — `FakeLLM.structured(**kwargs)`. Принимает **любые** kwargs и потому физически не может поймать «форму запроса». В коммите `f295154` написано прямым текстом: «фейки с `**kwargs` её не ловили».

### Слой 1 — фейк на транспортном уровне [strong]
`anthropic` и `openai` SDK ходят через `httpx`. Подменять надо `httpx`, а не Python-объект: `respx` (≈0.22) или `pytest-httpx` — вы видите **реальное тело JSON**.

На этом слое — **request-shape validator**, проверяющий инварианты протокола:
- на каждый `tool_use` в ходе ассистента в следующем user-ходе есть `tool_result` с тем же `tool_use_id`;
- роли чередуются, первый ход — `user`;
- нет `temperature`/`top_p`/`top_k`, если модель их не принимает;
- `tools[].type` совпадает с зафиксированной константой версии;
- `max_tokens` в допустимом диапазоне.

Это «contract-verifying test double»: фейк, который **отказывает на неправильном запросе**. Его отсутствие пропустило оба живых бага.

### Слой 2 — записанные кассеты [suggestive]
`pytest-recording` поверх `VCR.py`: один живой прогон каждого бэкенда (`--record-mode=once`), фильтр заголовков (`x-api-key`, `authorization`), YAML в git. Ловит дрейф SDK и провайдера. Оговорка: стриминг (SSE) VCR.py пишет сырым телом — кассеты большие и хрупкие. Для Claude Agent SDK неприменимо (подпроцесс, не HTTP) — нужен фейковый `claude` в `PATH` либо подмена `query()`.

### Слой 3 — бесплатная живая проверка формы [strong]
Endpoint `/v1/messages/count_tokens` принимает **то же тело**, что и `messages.create` (`model`, `system`, `messages`, `tools`), и не тарифицируется по токенам генерации. Готовый «контрактный смоук»: собрать реальный запрос для каждой роли, отправить, ожидать 200. Ловит `temperature → 400`, устаревшую версию `web_search_2026xxxx`, кривую схему инструмента — **до** платного прогона. `@pytest.mark.live`, вручную/ночью. **Самая недооценённая вещь в отчёте: полная защита от «malformed request» почти бесплатно.**

### Golden/snapshot тесты промптов [suggestive]
`syrupy` (`--snapshot-update`): на фиксированном `RunState` рендерить полный system+user каждой роли и снапшотить. **Мета-луп правит промпты автоматически** — любая правка даёт ревьюабельный diff.

### Property-based (Hypothesis) [strong для инструмента]
Большой пласт *чистой* логики — `check_stop`, `select_assumptions`, `apply_judge_verdict`, `gate.py`, `metrics.aggregate`. Инварианты:
- `select_assumptions` никогда не возвращает допущение в `CLOSED_STATUSES` и не длиннее `limit`;
- `aggregate` инвариантен к перестановке входов;
- Gate: либо `promote=False`, либо число сравнений ≥ порога;
- `apply_judge_verdict`: `consecutive_rollbacks` монотонно растёт до отката и обнуляется при принятии.

### Mutation testing [suggestive]
`mutmut` v3 (детект-рейт ~88.5% против ~82.7% у Cosmic Ray). Только по `gate.py`, `metrics.py`, `orchestrator.py`. 524 теста почти наверняка дадут выживших мутантов в `gate.py`: у порогов есть тесты «выше»/«ниже», но нет теста на границу и на вырожденную выборку.

### Fault injection — категория, которой нет вовсе [strong]
`FaultyBackend`-декоратор со сценариями: `RateLimitError` с `retry-after`, `APIConnectionError`, 529 overloaded, `ResponseTruncated`, ответ без `tool_use`, невалидный JSON, пустой текст, отказ модели, зависание. Табличный тест «роль × отказ» с проверками: (1) повторили ли в соответствии с `RETRYABLE_ERRORS`; (2) `state.json` после падения **загружается** и `resume` доходит до конца; (3) частичной записи нет; (4) usage не удвоился.

**Живой дефект [strong].** `RETRYABLE_ERRORS` (`orchestrator.py:47`) содержит только `anthropic.APIConnectionError`, `anthropic.RateLimitError`, `anthropic.InternalServerError` и `ValueError`. Исключения `openai` **не наследуются** от anthropic'овских. **429 от DeepSeek или OpenAI не ретраится вообще и убивает многоминутный прогон с первой попытки.**

**Второй дефект того же класса [strong].** `_with_retry` (`orchestrator.py:401`) — экспоненциальная пауза **без jitter** и без учёта `retry-after`. При `ThreadPoolExecutor` все потоки, получившие 429 в одну секунду, повторят строго через 2.0 с — thundering herd.

### Метаморфический тест на resume [suggestive, высокая отдача]
Детерминированный скриптовый бэкенд + «убить после шага k» для каждого k → `resume` → итоговый `RunState` совпадает с непрерывным прогоном с точностью до таймстемпов. Для системы, у которой возобновляемость — заявленная фича, это **самый ценный один тест в списке**.

## 3. Eval-харнессы: брать или не брать

| Инструмент | Что хорошо | Цена интеграции | Годится |
|---|---|---|---|
| **inspect-ai** (UK AISI, MIT) | dataset → Task → Solver → Scorer; sandbox, лог-формат `.eval`, VS Code viewer; провайдер `mockllm` для офлайна | Средняя | Единственный реальный кандидат. Риск: **своя** абстракция провайдеров, в которую не влезает подписочный CLI |
| **promptfoo** (MIT, YAML) | Регрессии промптов в CI, red-team, локально; куплен OpenAI в марте 2026 [suggestive] | Тянет Node-тулчейн в Python | Нет. Единица сравнения — многоминутный пайплайн, не промпт |
| **DeepEval** | 50+ метрик, `assert_test()` | Низкая технически | Нет. Метрики — LLM-судьи (деньги и ключ), а метрики Kaidzen детерминированы. По умолчанию телеметрия |
| **Braintrust / LangSmith / Weave** | Полный цикл, UI | Аккаунт | Дисквалифицированы: lock-in в офлайн OSS-библиотеке |
| **Phoenix / Arize** (Apache-2) | OTel/OpenInference, air-gapped | Это сервер, не библиотека | Как *опциональный* приёмник трейсов — да |
| **Ragas** | Метрики RAG | Низкая | Нет; требует LLM-судью |
| **OpenAI Evals** | — | — | Legacy; официальный cookbook посвящён миграции на promptfoo |

**Рекомендация: не брать ни одного как зависимость.** Харнесс даёт четыре вещи — датасет, скореры, ранер, хранилище. Все четыре уже есть: `benchmark/ideas` (детерминированный train/holdout), `metrics.py` (скореры **кодом**, а не моделью — сильнее, чем у любого из списка), `evolve.py` (ранер с кэшем и возобновлением), `evolution_log.py`. Взять харнесс — отдать ему цикл прогона и слой моделей, выбросив подписочный бэкенд.

**Правильный ход — делать себя читаемым для них, а не переезжать:** писать `events.jsonl` рядом со `state.json` и опциональный OTLP-экспорт. Если начальство требует «взять один» — **inspect-ai**.

## 4. Стоимость и латентность

### Prompt caching, Anthropic — точные механики [strong]
- До **4 breakpoint'ов** на запрос; `cache_control: {type: "ephemeral"}` либо один top-level для автоматического режима.
- Иерархия префикса: `tools` → `system` → `messages`. Изменение уровня инвалидирует его и всё после.
- Минимум для кэширования: **1024 токена** для Sonnet 5 / Opus 4.8, **512** для Opus 5 / Fable 5 / Mythos 5, **4096** для Opus 4.6/4.5 и Haiku 4.5. Короче — **не кэшируется, без ошибки**.
- TTL: 5 мин (умолчание) и 1 час. Множители: запись 5m — **1.25×**, запись 1h — **2×**, чтение — **0.1×**. Окупаемость: 5m после одного чтения, 1h после двух.
- Инвалидирует: изменение `tools` (роняет всё), переключение web search (system+messages), `tool_choice` (messages), `thinking`/`effort`, добавление/удаление изображений.
- **Время генерации входит в TTL**: «если ответ стримился 4 минуты, следующий запрос по тому же префиксу должен стартовать в пределах ~1 минуты». Для Researcher с 16k `max_tokens` и adaptive thinking **5-минутный TTL может не дожить до следующего хода** → для роли с поиском брать `ttl: "1h"`.
- Параллельные запросы: запись доступна только после начала первого ответа. При `ThreadPoolExecutor` все потоки промахнутся и каждый заплатит 1.25×. Лечится прогревом.
- `usage` возвращает `cache_creation_input_tokens` и `cache_read_input_tokens`; `total_input = cache_read + cache_creation + input_tokens`.

### Дефект учёта [strong, доказано данными репозитория]
`anthropic_api._record_usage` суммирует только `response.usage.input_tokens`. `claude_agent._record_usage_dict` — только `usage.get("input_tokens")`. Оба **игнорируют** `cache_creation_input_tokens` и `cache_read_input_tokens`. Claude Agent SDK кэширует агрессивно → почти весь вход приходит в кэш-полях. Доказательство в прогонах:
```
runs/2026-08-30-224740-baraholka-v2: input_tokens=102, output_tokens=111789, web_searches=44
runs/2026-08-30-205010-baraholka:    input_tokens=68,  output_tokens=52442,  web_searches=34
```
68 входных токенов на 34 веб-поиска физически невозможно. **Учёт входа сломан.** Следствие: `Gate.MAX_COST_GROWTH` считает удорожание **только по выходным токенам** → мутация, раздувшая системный промпт вдвое, пройдёт Gate как «не подорожавшая».

### Batch API [strong]
50% скидки на input и output. Лимиты: 100 000 запросов / 256 МБ на батч; большинство завершается менее чем за час, потолок 24 часа. `max_tokens: 0` внутри батча запрещён. Документация советует 1-часовой TTL кэша при батчах с общим контекстом. Множители кэша и скидка батча **стекаются**.

Внутри одного прогона роли последовательны — батчить нечего. Но эволюция — это `кандидаты × идеи` независимых прогонов; можно исполнять «волной» (wavefront). Отдача 50%, трудоёмкость высокая. Дешёвый частный случай: **слепые попарные сравнения отчётов** embarrassingly parallel — батчатся почти без изменений.

### Атрибуция по ролям — сейчас невозможна [strong]
`registry.build_backends` создаёт **один экземпляр на имя бэкенда**, а все четыре роли смотрят на `subscription`. `backend.usage` — общий счётчик четырёх ролей, `state.api_usage` — одно число на прогон. **Без атрибуции разговор об оптимизации стоимости беспредметен.** Нужно вернуть usage из вызова (`structured() -> (T, CallUsage)`) либо передавать метку роли и копить `usage_by_step`.

## 5. Детерминизм и воспроизводимость

**Почему воспроизводимости не будет даже при temperature=0 [strong].** Причина — **зависимость ядер от размера батча**: разные размеры запускают разные стратегии редукции, порядок суммирования меняется, результат меняется в последнем бите и распространяется на выбор токена. Отдельные ядра детерминированы, forward pass детерминирован, сервер детерминирован — а пользователь видит недетерминизм, потому что нагрузка (размер батча) меняется непредсказуемо. Полная воспроизводимость требует batch-invariant RMSNorm, matmul и attention ценой ~20% скорости — только на своём инференсе.

Плюс: Anthropic не даёт `seed` вообще, `claude-sonnet-5` возвращает 400 на любой sampling-параметр (поймано в `f295154`). У OpenAI `seed` — best-effort. **Не гоняйтесь за детерминизмом выхода — гоняйтесь за детерминизмом входа и провенансом.**

### Run manifest [suggestive]
Рядом со `state.json` — `manifest.json`:
```
kaidzen_version, git_sha
candidate_id, candidate_dir
prompt_hashes: {analyzer: sha256, researcher: …, refiner: …, judge: …}
config_hash, schema_hashes: {ResearcherOutput: sha256(model_json_schema()), …}
backends: {name: {type, sdk_version, model_requested, model_returned}}
tool_versions: {web_search: "web_search_20260209"}
effort, max_tokens, started_at, finished_at
```
Закрывает существующую дыру: `resume` читает промпты **с диска** и не проверяет, что они те же, что были при старте. **Мета-луп правит промпты автоматически. Прогон, прерванный до мутации и возобновлённый после, молча смешает два кандидата.** Хэш промптов + сверка при resume — десять строк.

### Content-addressed кэш вызовов ролей [suggestive]
Ключ = `sha256(canonical_json({backend_type, model, effort, system, user, schema_json, tool_versions}))`, значение = `{validated_output_json, usage, raw_response, created_at}`, хранилище — `.kaidzen-cache/<ab>/<key>.json`.

Сейчас checkpoint на уровне *шага* внутри прогона. Content-addressed кэш — на уровне *вызова*, живёт **между** прогонами и кандидатами. Resume становится частным случаем. В эволюции важнее всего: мутация правит **один** промпт роли, значит все вызовы до этой роли на первой итерации байт-в-байт совпадают у чемпиона и челленджера.

Оговорки: (1) вызовы с веб-поиском зависят от времени — исключать `researcher` или давать TTL и `--refresh`; (2) при попадании в кэш **не приплюсовывать** usage повторно.

## 6. Наблюдаемость

**OpenTelemetry GenAI semantic conventions [strong].** Соглашения `gen_ai.*` вынесены в `open-telemetry/semantic-conventions-genai` (v1.42.0, 12 июня 2026), стабильность — **Development**, 1.0 нет. Стабильны только `error.type` и server-атрибуты. Имена ключей брать оттуда, но не строить жёстких контрактов.

- Имя спана: `{gen_ai.operation.name} {gen_ai.request.model}`, напр. `chat claude-sonnet-5`.
- Обязательные: `gen_ai.operation.name` (`chat`, `invoke_agent`, `execute_tool`), `gen_ai.provider.name`.
- Условно обязательные: `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`.
- Opt-in: `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions`.

Иерархия: `invoke_agent kaidzen.pipeline` → спан итерации → `chat <model>` на роль → `execute_tool web_search`. Свои атрибуты: `kaidzen.run_id`, `kaidzen.candidate_id`, `kaidzen.iteration`, `kaidzen.step`, `kaidzen.generation`, `kaidzen.cache_hit`, `kaidzen.attempt`.

**Не тащить OTel в core [suggestive].** Библиотека, обещающая «отчёт собирается без обращения к модели», не должна требовать SDK телеметрии. Писать `events.jsonl` (одна JSON-строка на событие, атомарный append), экспортёр JSONL→OTLP — в `kaidzen[otel]`.

**Что записывать, чтобы разобрать упавшее поколение без повторного прогона [strong — прямой пробел].** Сейчас при `SchemaValidationFailure` наверх уходит `str(last_error)`, а **сырой текст ответа выбрасывается**. Минимум:
1. Полный запрос (system + user + tools), с редакцией ключей.
2. **Сырой ответ, не прошедший валидацию** — в `run_dir/failures/<step>-<attempt>.txt`.
3. `stop_reason` и `finish_reason`.
4. Полный `usage`, включая кэш-поля.
5. Поисковые запросы и URL результатов (сейчас считается только *количество*).
6. Номер попытки, задержка, класс исключения, цепочка `__cause__`.
7. Wall-clock на вызов (диагностика TTL кэша и таймаутов).

## 7. Абстракция провайдеров

**Вердикт: свой registry — правильное решение, и подписочный бэкенд именно поэтому [strong].**
LiteLLM, Instructor и Pydantic-AI одинаково моделируют «провайдера» как HTTP-endpoint + ключ + OpenAI-подобные параметры. `claude_agent_sdk` — локальный процесс: без ключа, без `max_tokens`, без `temperature`, без `seed`, с другим набором возможностей (встроенный WebSearch, `max_turns` вместо `max_searches`). Слота для него нет ни в одной из трёх.

- **Pydantic-AI** — ближайший по духу (есть `Model` ABC, output validators, retries, OTel), но тянет весь фреймворк ради одного метода.
- **LiteLLM** — тяжёлая зависимость с частыми ломающими изменениями; настоящая ценность — **таблица цен** (`litellm.model_cost` / `completion_cost`) или более лёгкий `tokencost`. Брать только таблицу.
- **Claude Agent SDK как единая абстракция** — нет: не умеет OpenAI/DeepSeek.

Что улучшить, не меняя решения:
1. `structured()` уже протекает: `effort`, `max_searches`, `max_tokens` игнорируются двумя бэкендами из трёх. Сделать явным: `Capabilities` — `supports_effort`, `supports_max_tokens`, `supports_seed`. Валидатор конфига уже читает `supports_web_search` до платного вызова — расширить механизм.
2. Вынести общий retry-цикл в базовый класс.
3. Вернуть usage из вызова.

## Какие из четырёх живых багов это ловит

| Живой баг | Фикс | Что поймало бы **до** прогона |
|---|---|---|
| **Сломанный протокол ретрая** — дописывался только новый user-месседж, без хода ассистента и `tool_result`; терялись результаты поиска | `2745fa3` | 1) `respx` + инвариант «на каждый `tool_use` есть `tool_result` с тем же id» — фейк на `**kwargs` этого не видит физически. 2) Fault-injection: провалить валидацию и проверить *второй исходящий запрос*. 3) **Дефект сейчас живёт в `claude_agent.py` и `openai_compat.py`.** 4) Одна реализация вместо трёх убирает класс целиком |
| **Кривая форма запроса** — `temperature` на `claude-sonnet-5` → 400; `max_tokens` 4096 при adaptive thinking; устаревший `web_search_20250305` | `f295154` | 1) Смоук на `/v1/messages/count_tokens` — бесплатно ловит все три. 2) `respx` + валидация тела против типов SDK. 3) Snapshot исходящего запроса (`syrupy`). 4) Матрица «модель × допустимые параметры» как данные |
| **Коллизии имён между прогонами** | `61a66b3`, `db22cad`, `0a67417` | 1) Hypothesis: идентификаторы попарно различны для любых `(generation, attempt, evolve_id)`. 2) Конкурентный тест: N параллельных стартов → N каталогов. 3) `mkdir(exist_ok=False)` — коллизия падает громко. 4) Property на обход каталогов |
| **Схлопнувшаяся выборка сравнения** — два падения чемпиона свели пять идей к одному сравнению | `2da09fc` | 1) **Инвариант в `gate.py`**, а не в ранере: `len(общие идеи) >= MIN_COMPARISONS`, иначе `promote=False`. Сейчас Gate доверяет тому, что принесли. 2) Hypothesis: при *любом* паттерне падений Gate никогда не продвигает на одном сравнении. 3) Fault injection в eval-пул. 4) Mutation testing по `gate.py` |

**Плюс два дефекта того же класса, ещё не пойманных живым прогоном:**
- **[strong]** `openai.RateLimitError` / `openai.APIConnectionError` не в `RETRYABLE_ERRORS` → 429 от DeepSeek/OpenAI убивает прогон без единой повторной попытки.
- **[strong]** Учёт входных токенов игнорирует кэш-поля (доказано данными). `MAX_COST_GROWTH` меряет удорожание только по выходу.

## Модель стоимости

Расчёт на **Claude Sonnet 5** ($2/MTok in, $10/MTok out, cache read $0.20, 5m write $2.50, 1h write $4.00; web search $10/1000). База — реальный прогон `2026-08-30-224740-baraholka-v2`: 4 итерации, 111 789 выходных токенов, 44 поиска, 13 вызовов ролей.

| Статья | Оценка | Доля |
|---|---|---|
| **Повторная отправка контекста в tool-loop Researcher'а** — каждый ход пересылает весь диалог, результаты поиска копятся; ~10 ходов × растущий префикс ≈ 400k входных **на один вызов**, ×4 итерации ≈ 1.6M | **$3.20** | **65%** |
| Выходные токены (13 × ~8.6k с thinking) | $1.12 | 23% |
| Веб-поиск, 44 × $0.01 | $0.44 | 9% |
| Вход остальных 9 вызовов | $0.18 | 4% |
| **Итого за прогон** | **≈ $4.94** | |
| **Поколение эволюции** (~10 свежих прогонов + мета-роли) | **≈ $50–60** | |

**Деньги не в генерации, а в квадратичной пересылке контекста внутри агентного цикла с поиском.** [speculative — реальный вход не измеряется, см. дефект учёта]

### Топ-3 экономии

**1. Prompt caching на растущем префиксе — минус ~45–50% [strong по механике, speculative по величине].**
`cache_control` на последнем блоке перед каждым новым ходом. Уникальных токенов в вызове Researcher'а ≈ 69k; пишутся один раз (1.25×) и читаются ~9 раз по 0.1× вместо 410k по 1.0×.
- Было: 410k × $2/M = **$0.82** за вызов Researcher'а.
- Стало: ≈ 118k-эквивалента × $2/M = **$0.24**.
- На прогон: **$3.28 → $0.94**, итог **$4.94 → $2.60**.
Условия: TTL `1h` для ролей с поиском; прогрев одним запросом перед пулом; минимум 1024 токена (системный промпт 6–9 КБ проходит); следить, что `effort` и web search не меняются внутри роли. Трудоёмкость — часы.

**2. Content-addressed кэш вызовов — минус 10–25% поколения [speculative].**
Мутация правит один промпт. Всё до неё на первой итерации идентично побайтно. Если мутирован `judge` — бесплатны analyzer + researcher + refiner первой итерации (~20% прогона). Матожидание ≈ 10–25% поколения, **$5–15**. Побочная выгода важнее: повтор неизменённого шага после падения бесплатен.

**3. Batch API на веере эволюции — минус 50% [strong по ставке, high по трудоёмкости].**
Стекается с кэшем. Эволюция офлайновая, 24 часа приемлемы. Требует wavefront-исполнения — переписывание оркестратора. **$50 → $25 за поколение**, но недели работы. Дешёвый частный случай первым: **слепые попарки** батчатся почти без изменений.

*Почётное четвёртое:* дедуп поисковых запросов между итерациями (~5% прогона); маршрутизация Judge/слепого судьи на Haiku 4.5 ($1/$5 против $5/$25 у Opus 5) — 5× на роли, риск измеряется самим Gate.

## Ранжированные рекомендации

| # | Рекомендация | Влияние | Труд | Риск |
|---|---|---|---|---|
| 1 | Чинить учёт токенов: `cache_creation_input_tokens` + `cache_read_input_tokens`; вернуть usage из вызова, копить `usage_by_step` | **Очень высокое** | Низкий | Низкий |
| 2 | Смоук формы запроса через `/v1/messages/count_tokens` на роль и бэкенд | **Очень высокое** | Низкий | Низкий |
| 3 | `openai.*` в `RETRYABLE_ERRORS`; full jitter и `Retry-After` | Высокое | Низкий | Низкий |
| 4 | Prompt caching: `cache_control` на префиксе, `ttl:"1h"` для ролей с поиском, прогрев перед пулом | **Очень высокое** | Низкий-средний | Средний |
| 5 | Сохранять сырой невалидный ответ в `run_dir/failures/` + `events.jsonl` | Высокое | Низкий | Низкий |
| 6 | `json-repair` вместо `_slice_outer_braces`; проверка `stop_reason` в `claude_agent` и `openai_compat` | Высокое | Низкий | Низкий |
| 7 | Retry-with-error-feedback в базовый класс: три копии → одна | Высокое | Средний | Средний |
| 8 | `respx` + request-shape validator вместо `**kwargs`-фейков | **Очень высокое** | Средний | Низкий |
| 9 | Fault-injection матрица «роль × бэкенд × отказ» с проверкой возобновляемости | Очень высокое | Средний | Низкий |
| 10 | Инвариант минимальной выборки в `gate.py` + Hypothesis-property | Высокое | Низкий | Низкий |
| 11 | Run manifest с хэшами промптов/схем/конфига + сверка при `resume` | Высокое | Низкий | Низкий |
| 12 | Content-addressed кэш вызовов ролей | Высокое | Средний-высокий | Средний |
| 13 | Hypothesis на чистую логику | Среднее | Низкий | Низкий |
| 14 | Snapshot-тесты промптов (`syrupy`) | Среднее | Низкий | Низкий |
| 15 | Кассеты `pytest-recording`/VCR.py | Среднее | Средний | Средний |
| 16 | `mutmut` по `gate.py`/`metrics.py`/`orchestrator.py` | Среднее | Низкий | Низкий |
| 17 | Метаморфический тест resume «убить после шага k» | Высокое | Средний | Низкий |
| 18 | Явные `Capabilities` вместо молча игнорируемых параметров | Среднее | Низкий | Низкий |
| 19 | Опциональный `kaidzen[otel]`: `events.jsonl` → OTLP | Среднее | Средний | Низкий |
| 20 | Таблица цен из `litellm.model_cost` / `tokencost` | Среднее | Низкий | Низкий |
| 21 | Batch API для слепых попарок | Среднее | Средний | Средний |
| 22 | Batch API + wavefront для всего веера | Высокое | **Высокий** | Высокий |
| — | **Не делать:** eval-харнесс как зависимость; переезд на LiteLLM/Pydantic-AI/Instructor; Outlines/XGrammar/llguidance | — | — | — |

## Источники
- Prompt caching — https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Batch processing — https://platform.claude.com/docs/en/build-with-claude/batch-processing
- Pricing — https://platform.claude.com/docs/en/about-claude/pricing
- Token counting — https://platform.claude.com/docs/en/build-with-claude/token-counting
- Rate limits — https://platform.claude.com/docs/en/api/rate-limits
- OTel GenAI spans — https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md
- Defeating Nondeterminism in LLM Inference — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- llguidance — https://github.com/guidance-ai/llguidance · XGrammar — https://arxiv.org/pdf/2411.15100
- JSONSchemaBench — https://arxiv.org/html/2501.10868v1
- Instructor — https://python.useinstructor.com/
- pytest-recording — https://github.com/kiwicom/pytest-recording
- Hypothesis — https://hypothesis.readthedocs.io/
- mutmut vs Cosmic Ray (IEEE) — https://ieeexplore.ieee.org/document/10818231/
- Inspect AI — https://hamel.dev/notes/llm/evals/inspect.html
- Promptfoo CI/CD — https://www.promptfoo.dev/docs/integrations/ci-cd/
- Arize Phoenix — https://github.com/arize-ai/phoenix
- LiteLLM providers — https://docs.litellm.ai/docs/providers
