# Level 2 Kaidzen vs SOTA автоматической оптимизации промптов

## (a) Сравнительная таблица методов

| Метод | Механизм | Заявленный выигрыш | Стоимость | Зрелость | Годность для Kaidzen |
|---|---|---|---|---|---|
| **APE** (ICLR'23) | LLM генерирует кандидатов-инструкций из демонстраций, отбор по score, iterative Monte Carlo resampling | 24/24 Instruction Induction, 17/21 BBH на уровне человека [strong] | Десятки-сотни eval-прогонов | Исторический baseline | Низкая. Нет traces, нет multi-role |
| **OPRO** (ICLR'24) | В meta-prompt кладётся история `(prompt, score)`, LLM выдаёт следующий prompt. Только скаляры | до +8% GSM8K, до +50% BBH [strong] | Каждый шаг = полный eval; сотни шагов | Репродукции есть, в проде редко | Низкая. Kaidzen уже сильнее: диагност видит тексты отчётов |
| **ProTeGi / APO** (EMNLP'23) | Minibatch → LLM пишет «textual gradient» → правка; beam search + **multi-armed bandit** | Обгоняет прежние методы на 4 задачах [strong] | Дёшево (minibatch + bandit) | Академический | **Высокая** — bandit/beam именно то, чего нет |
| **TextGrad** (Nature 03.2025) | Autograd-аналог: граф LLM-вызовов, backprop текстового фидбэка | GPQA 51%→55%; LeetCode-Hard +20% rel [strong] | Backward на каждый узел = дорого | Nature, репо живое | Средняя. Требует переписать pipeline в граф |
| **DSPy BootstrapFewShot** | Teacher генерирует демонстрации | Часто сильнейший при малых данных [strong] | ~$2, ~10 мин, от ~10 примеров | Очень зрелый | Низкая: Kaidzen оптимизирует инструкции |
| **DSPy COPRO** | Coordinate ascent по инструкциям модулей | Скромный | от 50 примеров | Устарел против MIPROv2 | Низкая |
| **DSPy MIPROv2** | Bayesian/TPE-суррогат над (instruction × demo) для всех модулей; minibatch eval | +5.6% aggregate [strong] | 30 trials, minibatch 25, **~200+ примеров** | Индустриальный стандарт | Средняя: требует ≥200 идей |
| **GEPA** (ICLR'26 Oral) | Reflective mutation по **трассам одного модуля** + текстовый фидбэк метрики (не скаляр). Архив + **Pareto по задачам**. Round-robin модулей. System-aware merge | **+10% над GRPO** (до +19–20%) при **35× меньше rollouts**; **>2×** над MIPROv2; промпты на 33% короче [strong] | HotpotQA: 737 обучающих rollouts; **minibatch=3**; от ~3 примеров, реком. 20–100 | MIT, пакет `gepa`, интеграции | **Наивысшая.** Тот же контур + архив, Pareto, трассы |
| **PromptBreeder** (ICML'24) | Популяция 50, 20–30 поколений, tournament. **Self-referential**: мутируются mutation-prompts (hypermutation), Lamarckian, EDA, crossover 10% | Обгоняет CoT/Plan-and-Solve [strong] | 50 × 25 × полный eval — очень дорого | **Кода нет** | Идейно высокая, практически — источник операторов |
| **EvoPrompt** (ICLR'24) | GA/DE поверх популяции промптов | до +25% на BBH [strong] | pop=10, T=10, dev ≤200; early-stop −57…74% | Код открыт, поддержки мало | Средняя: доказывает что pop=10 достаточно, pop=2 — нет |
| **CAPO** (2025) | Популяция + **racing** из AutoML + штраф за длину | до +21 п.п.; лучше SOTA в 11/15 [strong] | Racing экономит основной бюджет | Свежий | **Высокая** — racing и length-penalty ложатся на Gate |
| **AdalFlow** | «PyTorch для LLM-workflow», textual gradients | Паритет с TextGrad/DSPy | ~TextGrad | Активен, мал | Низкая: дублирует GEPA без Pareto |
| **ADAS** | Мета-агент **пишет код агентов**, архив идёт ему на вход | Обгоняет hand-designed на ARC [strong] | Очень дорого | Research | Низкая, но **archive-as-context** = EVOLUTION-log |
| **AlphaEvolve** | Ансамбль Gemini, program database с **island-моделью** | 4×4 умножение за 48 скаляров (Штрассен 49); 0.7% compute Borg; +23% FlashAttention [strong] | Промышленный масштаб | Закрыт (OpenEvolve) | Источник паттерна island model |
| **Darwin-Gödel Machine** | Агент переписывает свой код; **архив всех агентов, не один чемпион** | SWE-bench 20%→50%, Polyglot 14.2%→30.7% [strong] | Огромная | Research; авторы сами описывают objective-hacking | Идейно высокая: **архив > single champion** |

## (b) Чего не хватает контуру Kaidzen

Сделано правильно, не ломать:
- **Двойное судейство с перестановкой позиций** (`_challenger_wins`: победа только при согласии обоих порядков). Position bias систематичен [strong]. Строже большинства.
- **Gate на объективных метриках поверх судьи**, `_gamed_the_denominator`, `_too_expensive` (>1.5×). CAPO вводит length penalty — здесь жёстче.
- **≤2 роли за мутацию** — ручная, но корректная форма credit assignment.
- **EVOLUTION-log с rejected-памятью** — archive-as-context из ADAS.

### Дыра №1 — бенчмарк из одной идеи [strong, критично]
`benchmark/business/ideas/` = **один файл**. `_holdout_size(1) = 0`:
- train = 1, holdout **пустой** — защита от переобучения выключена;
- `win_rate` по 1 паре ∈ {0.0, 1.0}, порог 0.55 вырождается в «судья сказал да»;
- `MIN_COMPARABLE_IDEAS = 2` не блокирует, только пишет в лог.

Даже при 6 идеях: под H0 (кандидат = шум, p=0.5) вероятность ≥4 побед из 6 = **34%**. Треть чистого шума проходит Gate. Литература: single-trial flip rate **13.6%**, для ~5% ошибок нужно **11 повторов** [strong]. DSPy: 10 примеров минимум, 50+ для GEPA/COPRO, 200+ для MIPROv2 [strong].

**Всё остальное вторично. Оптимизация с N=1 — не оптимизация.**

### Дыра №2 — единственный чемпион вместо архива [strong]
`state.champion_dir` — один указатель. Отвергнутый челленджер никогда не станет родителем. Hill-climbing по одной вершине.

- **GEPA**: родитель выбирается стохастически с **Pareto-фронта** по отдельным задачам. Абляция: **+6.4% aggregate, до +8.17%** против SelectBestCandidate [strong].
- **DGM**: явно ветвится от неоптимальных предков — так и получилось 20%→50%.
- **AlphaEvolve**: island model.

Блокер: `CandidateRecord.metrics` — агрегат по всем идеям. **Per-idea вектор счётов не сохраняется.**

### Дыра №3 — beam=2 без экономии на оценке [suggestive]
`CHALLENGERS_PER_GENERATION = 2`. Проблема не в ширине как таковой, а в том, что обе мутации **всегда от одного родителя** и обе оцениваются **полным** прогоном. Нет minibatch (GEPA=3, MIPROv2=25), нет bandit (ProTeGi), нет racing (CAPO). `stop_on_failures` отсекает только падения, не плохое качество.

### Дыра №4 — диагност работает по агрегатам, не по трассам [strong]
На входе `run_diagnostician`: агрегированные `RunMetrics`, финальные тексты отчётов, дайджест журнала.

Нет: по-ролевых промежуточных выходов; отклонённых/невалидных ответов и ошибок парсинга; per-idea разбивки; машиночитаемых `metrics_delta` прошлых поколений.

GEPA явно строит фидбэк-функцию μ_f, возвращающую **диагностический текст, а не скаляр** — «сообщения компилятора до схлопывания в reward» [strong]. Диагност Kaidzen видит уже схлопнутое. `diagnostician.md` компенсирует вручную зашитой каузальной эвристикой («низкий closed_rate + высокий partial_rate ⇒ Researcher») — то есть **человек сделал credit assignment за модель**, покрыв один случай из многих.

Мутатор ещё беднее: `run_mutator(champion, diagnosis, do_not_break)` — **не видит ни одного примера провала**.

### Дыра №5 — credit assignment между 5 ролями [strong]
Известно только `roles_touched`. MIPROv2 факторизует по модулям; GEPA выбирает модуль **round-robin**, атрибуция тривиальна по построению.

**Неиспользуемый бесплатный сигнал**: метрики уже раскладываются по ролям семантически — `high_total` → Analyzer, `closed_rate`/`partial_rate` → Researcher, `grounded_changelog_rate` → Refiner, расхождение рубрики с метриками → Judge. Эта карта прописана прозой в `diagnostician.md`. Должна быть кодом.

### Дыра №6 — переобучение измеряется поздно и грубо [strong]
`CHECKPOINT_EVERY = 3`, holdout только на чекпоинте, сейчас не гоняется вообще.
- OPRO: train выше test на **5–20%**; лечится большим train + early stopping [strong].
- GA-оптимизация: больший pop → больший generalization gap (0.22) [suggestive].
- Документация GEPA: без valset используется trainset и для обучения, и для отбора — **«это вызывает переобучение»** [strong].
- 10 validation-инстансов, p=0.5 → дисперсия 0.025, шум доминирует [strong].
- Богатый текстовый фидбэк **сам провоцирует** переобучение: модель зашивает конкретные примеры в промпт [suggestive]. Прямой риск: мутатор получает полные отчёты и может вписать особенности единственной бизнес-идеи.
- Утешительное: GEPA сообщает **меньший gap для instruction-оптимизации, чем для few-shot** [strong] — Kaidzen в лучшем режиме.

### Дыра №7 — нет crossover и мутации мета-промптов [suggestive]
- **GEPA merge**: собрать потомка из лучших версий разных модулей. До **+5%**, ≤5 раз за прогон [strong]. Кандидат = папка из 5 файлов, слияние = `shutil.copy`.
- **PromptBreeder hypermutation**: `prompts/meta/*.md` статичны. Level 2 эволюционирует Level 1, но не себя.

## (c) Брать библиотеку или писать своё

### DSPy (MIPROv2) — **нет** [strong]
1. Требует переписать pipeline в `dspy.Module`+`Signature`. Промпты Kaidzen — авторские русскоязычные `.md` со структурой, которая в модель DSPy не ложится.
2. **200+ примеров**. У Kaidzen 1.
3. Backends: `dspy.BaseLM` подклассить можно (3.2.x отвязывает от litellm: `forward_contract="typed_lm"`), но это работа против движущейся мишени — открыт эпик #9514.
4. Оптимизирует инструкции **и few-shot демо**; демо бессмысленны (каждая идея — длинный документ).

Потеряно: контроль над текстом промптов, `do_not_break`-семантика, Gate на доменных метриках. Приобретено: Bayesian surrogate, которому при N=1 нечего есть.

### `gepa` как standalone — **да, частично** [strong]
Ключевой факт: **`gepa` не требует DSPy**. `gepa-ai/gepa`, **MIT**:

```
gepa.optimize(seed_candidate, trainset, valset, task_lm, max_metric_calls, reflection_lm)
```
- `seed_candidate` — **словарь именованных текстовых компонент** = структура кандидата Kaidzen один-в-один.
- `reflection_lm` — обычный text-in/text-out, без протокола. Claude CLI-драйвер подключается обёрткой.
- `optimize_anything(...)` вообще не требует адаптера — достаточно функции-оценщика.
- Работает от 3 примеров, бюджет 100–500 метрик-вызовов.

Потеряно при полной замене: Gate на доменных метриках (у GEPA один скаляр + текст); слепые попарки с перестановкой (GEPA оценивает абсолютно); `EVOLUTION-<domain>.json` с `do_not_break`; checkpoint с человеком, resume, атомарность; обёртки мультибэкендности.

**Рекомендация: не заменять контур, а украсть три механизма; `gepa` держать открытым как опциональный второй движок.**
1. Инфраструктура Kaidzen (resume, чекпоинты, атомарные кандидаты, доменный Gate) — не то, что решает GEPA. Она решает *поиск*, а дефицит именно там.
2. Дефицит поиска = per-idea счета + Pareto + трассы + minibatch/racing. Каждое реализуется в существующей архитектуре за десятки строк — кандидат уже иммутабелен и лежит отдельной папкой.
3. Главное узкое место (1 идея) библиотекой не лечится.
4. `optimize_anything` как `--engine=gepa` **позже**, при ≥20 идеях — сравнить свой контур с эталонным на одном бенчмарке.

## (d) Ранжированные изменения

| # | Изменение | Влияние | Труд | Риск |
|---|---|---|---|---|
| **1** | **Бенчмарк до 12–20 идей на домен** (train ~10–14, holdout ~4–6). При 6 идеях 34% случайных кандидатов проходят порог. Ориентир GEPA — 20–100 [strong] | Критическое | Средний | Нулевой |
| **2** | **Жёсткий отказ при `len(train) < MIN_COMPARABLE_IDEAS`**. Сейчас `_compare` пишет в лог и продолжает; `_holdout_size` молча даёт 0. Нужно исключение как `BenchmarkEmpty` | Критическое | 30 мин | Нулевой |
| **3** | **Per-idea счёт в `CandidateRecord`** (`scores: dict[idea, float]`). Разблокирует 4, 6, 9 | Высокое | Малый | Низкий |
| **4** | **Pareto-выбор родителя вместо чемпиона.** Пул = прошедшие Gate или выигравшие хотя бы одну идею; отсечь доминируемых; сэмплить ∝ числу выигранных идей. GEPA: **+6.4%, до +8.17%** [strong] | Высокое | Средний | Средний — меняет семантику `champion_id`, нужен отдельный «продакшн-набор» |
| **5** | **Трассы диагносту и мутатору вместо агрегатов**: per-role выходы для 1–2 худших идей; допущения с вердиктом `partial` вместе с обоснованием; ошибки парсинга/отклонённые ответы; структурированные `metrics_delta` за 3 поколения. Мутатор сейчас **не видит ни одного провала** — главный разрыв с reflective gradient | Высокое | Средний | Средний — рост контекста; ограничить top-2 идеями |
| **6** | **Minibatch + racing.** Сначала 3 идеи (GEPA), продолжать при `win_rate ≥ 0.5`. CAPO: racing даёт основную экономию [strong]; EvoPrompt: −57…74% [suggestive]. `_evaluate_candidate` уже умеет `stop_on_failures`, нужен `stop_on_quality` | Высокое | Малый | Низкий — сэмплировать minibatch, а не брать первые 3 по имени |
| **7** | **Per-role scoreboard кодом**, а не прозой в `diagnostician.md`. Таблица «роль → метрики → дельта к прошлому поколению» | Высокое | Малый | Низкий |
| **8** | **Holdout каждое поколение** для промотированного, а не раз в 3. Ожидаемый gap 5–20% [strong]; при `CHECKPOINT_EVERY=3` подгонка закрепляется на три поколения | Среднее | Малый | Низкий (+1 прогон/поколение) |
| **9** | **System-aware merge (crossover)** между линиями, тронувшими разные роли. GEPA: до **+5%**, ≤5 раз [strong]. Слияние тривиально | Среднее | Малый | Средний — может нарушить неявные связки; обязателен полный прогон + Gate |
| **10** | **Повторы судейства при близком результате.** Сейчас 2 вызова с требованием согласия; при flip rate 13.6% для ~5% ошибок нужно ~11 [strong]. Компромисс: эскалация до 3×2 только там, где первые два не сошлись | Среднее | Малый | Низкий |
| **11** | **Length/cost penalty внутрь целевой функции**, а не только вето Gate. CAPO делает штраф частью objective — мутатор оптимизирует краткость, а не упирается в стену. GEPA-промпты на 33% короче [strong] | Среднее | Малый | Низкий |
| **12** | **Hypermutation мета-промптов** (PromptBreeder) раз в K поколений, оценка по доле промотированных потомков | Низкое сейчас | Средний | **Высокое** — вторая петля поверх шумной первой; только после 1–8 |
| **13** | **Опциональный `--engine=gepa`** через `optimize_anything` как эталон для сравнения | Низкое сейчас | Средний | Низкий |

**Порядок:** 1 → 2 → 3 → 5 → 7 → 6 → 8 → 4 → 10 → 11 → 9 → (13) → (12).

**Не трогать:** двойное судейство с перестановкой, `_gamed_the_denominator`, `MAX_COST_GROWTH`, «≤2 роли на мутацию», иммутабельность кандидатов, чекпоинты с человеком.

## (e) Источники

- GEPA (ICLR 2026 Oral) — https://arxiv.org/abs/2507.19457 · https://arxiv.org/html/2507.19457v1
- GEPA библиотека (MIT) — https://github.com/gepa-ai/gepa · https://gepa-ai.github.io/gepa/guides/adapters/
- DSPy GEPA overview — https://dspy.ai/api/optimizers/GEPA/overview/
- DSPy optimizers — https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md
- MIPROv2 — https://dspy.ai/api/optimizers/MIPROv2/ · https://arxiv.org/abs/2406.11695
- DSPy BaseLM — https://dspy.ai/api/models/BaseLM/ · эпик: https://github.com/stanfordnlp/dspy/issues/9514
- TextGrad (Nature 2025) — https://arxiv.org/abs/2406.07496 · https://github.com/zou-group/textgrad
- OPRO — https://arxiv.org/abs/2309.03409
- ProTeGi / APO — https://aclanthology.org/2023.emnlp-main.494/ · https://arxiv.org/abs/2305.03495
- APE — https://arxiv.org/pdf/2211.01910
- PromptBreeder — https://arxiv.org/abs/2309.16797
- EvoPrompt — https://arxiv.org/abs/2309.08532 · https://github.com/beeevita/EvoPrompt
- CAPO — https://arxiv.org/abs/2504.16005
- AdalFlow — https://arxiv.org/html/2501.16673v1 · https://github.com/SylphAI-Inc/AdalFlow
- ADAS — https://arxiv.org/abs/2408.08435
- AlphaEvolve — https://arxiv.org/abs/2506.13131
- Darwin-Gödel Machine — https://arxiv.org/abs/2505.22954 · https://sakana.ai/dgm/
- Position bias судей — https://arxiv.org/html/2406.07791v6
- Обобщение промптов — https://arxiv.org/html/2510.08413
- GAAPO (gap растёт с pop) — https://arxiv.org/pdf/2504.07157
