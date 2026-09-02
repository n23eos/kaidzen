# Factual grounding, attribution и citation verification для роли Researcher

Метки: **[strong]** подтверждено первоисточником · **[suggestive]** вторичные источники · **[speculative]** экстраполяция.

## (a) Inventory метрик attribution

| Метрика | Что измеряет | Как считается | Без LLM? | Стоимость |
|---|---|---|---|---|
| **ALCE** citation recall/precision | Подкреплён ли sentence цитируемыми документами, нужна ли каждая цитата | Recall: конкатенация cited docs → entails ли sentence. Precision: leave-one-out | **Да**, если NLI локальная (оригинал: TRUE = T5-11B) | κ с людьми 0.698 recall / 0.525 precision **[strong]** |
| **AutoAIS / TRUE** | Factual consistency (premise ⊨ hypothesis) | T5-11B на пуле NLI-датасетов | Да | 11B тяжело; заменяется MiniCheck/HHEM **[strong]** |
| **FActScore** | Доля atomic facts, подтверждённых источником | Decompose → retrieve → verify | **Нет** — декомпозиция требует LLM | 26–41 фактов на генерацию; human-eval $4/генерация **[strong]**. Есть OpenFActScore |
| **SAFE** (DeepMind) | Long-form factuality через Google Search | Decompose → self-contained rewrite → relevance filter → search-verify; метрика F1@K | **Нет** | 72% согласия с краудворкерами, выигрывает 76% споров, >20× дешевле людей **[strong]** |
| **RAGAS faithfulness** | claims_supported / claims_total | LLM извлекает claims, LLM судит | **Нет** | Есть класс non-LLM метрик: StringPresence, ExactMatch, BLEU, ROUGE, CHRF **[strong]** |
| **AttrScore** | attributable / extrapolatory / contradictory | Fine-tuned LM | **Да** | AttrScore-Flan-T5 (3B) ≈ fine-tuned GPT-3.5, лучше на OOD **[suggestive]** |
| **AttributionBench** | Насколько трудна автооценка attribution | Агрегация датасетов | — | **Потолок: даже fine-tuned GPT-3.5 ~80% macro-F1** **[strong]**. Ограничение на любую автометрику |
| **HHEM-2.1-open** (Vectara) | Hallucination score 0..1 для (premise, hypothesis) | Cross-encoder на FLAN-T5-base, **0.1B**, Apache 2.0, English-only, без лимита 512 токенов | **Да, полностью локально** | RAGTruth-QA balanced acc **74.28%** vs GPT-4 74.11%, GPT-3.5 56.16% **[strong]**. Самый дешёвый вариант |
| **MiniCheck** (EMNLP 2024) | Sentence-level grounding | Flan-T5/DeBERTa/RoBERTa на синтетике + NLI | **Да, локально** | GPT-4-level при **400× меньшей стоимости**. RoBERTa-L 355M, DeBERTa-v3-L 435M, Flan-T5-L 770M **[strong]** |
| **LLM-AggreFact** | Сравнение fact-checkers на 11 датасетах | Balanced accuracy | — | Bespoke-MiniCheck-7B **77.4**, Claude-3.5 Sonnet 77.2, gpt-4o **75.9**, FactCG-DeBERTa-L (0.4B) 75.6, MiniCheck-Flan-T5-L (0.8B) **75.0** **[strong]** |
| **DeepResearch Bench / FACT** | Citation Accuracy (C.Acc), Effective Citations (E.Cit) | C.Acc = доля пар «statement–URL» со статусом support | Нет | 100 PhD-задач, 22 области. Perplexity Deep Research — 90.24% C.Acc **[strong]** |

**Главный вывод:** метрика Kaidzen может остаться code-computed. 80% ценности дают **детерминированные проверки (URL, quote-substring), а не entailment**. Локальная NLI (HHEM, 0.1B) — второй слой, но её потолок ~75% balanced accuracy **[strong]**, поэтому она — *soft signal / triage*, а не жёсткий gate.

## (b) Конкретный citation-verification pipeline

Принцип из HALLMARK: **двухстадийный каскад «rule-based pre-screening + LLM diagnoser» даёт 0.996 detection при 0.108 FPR** **[strong]**. Детерминированное — первым, дорогое — только на эскалацию.

### Stage 0 — контракт вывода Researcher (без сети, без ML)
```
Verdict = {assumption_id, verdict, confidence: float,
           citations: [{url, quote, retrieved_at}],
           queries_issued: [str], results_seen: int}
```
Гейты: `len(queries_issued) == 0` → reject (уже есть); запросы были, но `citations == []` при `verdict != unverifiable` → reject; `quote` — **дословный фрагмент**, а не пересказ.

### Stage 1 — Transcript allow-list (offline, **самый сильный контроль**)
Ключевая находка. Kaidzen хранит сырые payload'ы search-тула. Тогда:
1. `url ∈ set(URLs, встретившихся в результатах search-тула)` — иначе URL **сфабрикован по построению**, без единого сетевого запроса.
2. `normalize(quote) ⊂ normalize(concat(tool_results_text))` — иначе цитата сфабрикована.

**100% precision** на детекции фабрикации (тождество, а не эвристика) и стоит ноль.

### Stage 2 — URL resolution (сеть, кэш)
Четырёхклассовая схема `urlhealth` (Rao, Wong, Callison-Burch, 2026) **[strong]**:
- `HEAD` (fallback `GET`) с browser-like UA, follow redirects, timeout, tenacity-retry.
- **LIVE** — 200. **DEAD (stale)** — 404 **и** есть снапшот в Wayback. **LIKELY_HALLUCINATED** — 404 **и** снапшота нет. **BLOCKED/UNKNOWN** — 403/429/connection error, **отдельная корзина, не фабрикация**.

Их измерения: 3–13% citation URLs галлюцинированы, 5–18% не резолвятся; Gemini-2.5-pro-deepresearch 13.3%, Claude 3.0–3.2%; deep-research агенты 10.7% против 4.8% у search-augmented LLM. Agentic self-correction снижает non-resolving в 6–79× **[strong]**.

DOI — отдельная ветка: `GET api.crossref.org/works/{doi}` (без ключа), 404 = не существует **[strong]**.

**Анти-ложные-срабатывания (HALLMARK failure mode i):** никогда не помечать FABRICATED по одному сигналу — при агентном lookup по нескольким БД это даёт ~5× больше FP **[strong]**. Конъюнкция: `not_in_transcript AND (404 AND no_wayback_snapshot)`.

### Stage 3 — Quote presence (offline после fetch)
1. `trafilatura.extract(html)` — лучший из открытых экстракторов: F1 ≈ 0.945–0.96 **[strong]**.
2. PDF → `pypdf`/`pdfminer.six`.
3. Нормализация: NFKC, схлопывание пробелов, lowercase, унификация кавычек/тире.
4. Точное вхождение → `exact`; иначе `rapidfuzz.fuzz.partial_ratio ≥ 90` → `fuzzy`; иначе `absent`.
5. Если extraction дал <200 символов или сработали paywall-маркеры → `page_unavailable` (**не** `absent`).

Бонус: ссылка с text fragment `#:~:text=<quote>` — читатель попадает прямо на подтверждающее место.

### Stage 4 — Entailment (локальная модель, опционально)
- **Default: HHEM-2.1-open** — FLAN-T5-base, 0.1B, Apache 2.0, произвольный контекст **[strong]**. На CPU без ускорителя **[speculative]**.
- **Strict: MiniCheck-Flan-T5-Large** (770M, 75.0 на LLM-AggreFact) **[strong]**. MiniCheck — **sentence-level**, многосоставный claim резать на предложения.
- Premise = окно ±N вокруг цитаты либо top-k пассажей по BM25 (`rank_bm25`). Hypothesis = утверждение из реестра в self-contained виде.

Поверх — **ALCE leave-one-out** для citation precision.

**В README записать:** потолок автоматической attribution-оценки ~80% macro-F1 **[strong]**, MiniCheck/HHEM ~75–77%. Entailment-score — метрика тренда и триаж, не приёмочный критерий.

### Stage 5 — агрегированные code-computed метрики
```
fabricated_url_rate        # Stage 1+2, цель 0
non_resolving_rate         # Stage 2
blocked_rate               # отдельно, не в числителе fabrication
quote_match_rate           # exact / fuzzy / absent / page_unavailable
citation_support_rate      # ~ ALCE citation recall
citation_precision         # leave-one-out
independent_domain_count   # per verdict
verdict_distribution + hedging_rate
risk_coverage_curve
```

**Декомпозиция claim'ов.** Протокол **Claimify** (ACL 2025): Sentence splitting → Selection (только verifiable propositions) → **Disambiguation (претензия отбрасывается при неоднозначной интерпретации)** → Decomposition **[strong]**. Стадия Disambiguation прямо решает проблему Kaidzen: не извлекать claim при неуверенности вместо того, чтобы извлечь и хеджировать.

## (c) Апгрейды поисковой стратегии

1. **Бюджет и разнообразие запросов.** Минимум 2, цель 3–5 различных на assumption. Проверять код-сайдом: Jaccard пересечения токенов между запросами < 0.6 — иначе три перифраза одного запроса «формально» проходят гейт.
2. **Обязательный disconfirming query.** Разложить на (a) подтверждающий, (b) **опровергающий** («X does not», «limitations of X», «criticism», «deprecated»), (c) primary-source (`site:`, `filetype:pdf`, doi/arxiv). (b) — самый дешёвый де-байасинг.
3. **Адаптивная остановка.** IRCoT не имеет стоп-критерия (жёсткий лимит шагов, фиксированные 15 параграфов); FLARE использует low-confidence токены чтобы решить *когда искать*, а не *когда остановиться*; TASR предлагает training-free adaptive stopping **[suggestive]**. Правило: стоп при ≥K **независимых доменов** согласны, **или** по saturation (новый запрос не принёс нового домена), **или** жёсткий cap.
4. **Считать независимые домены, а не URL.** `tldextract` → registrable domain. Две ссылки на один домен = один источник. Основа для «≥2 независимых источника → confirmed».
5. **Authority tiers + recency.** Тир по типу домена (standards/gov/edu/vendor-docs/repo > peer-reviewed > качественные news > блоги > content farms) **[suggestive]**. Для быстро меняющихся техно-утверждений — источник не старше ~18 месяцев, иначе `partial (temporal)`.
6. **Primary vs secondary.** Для утверждений про библиотеку/API — только vendor docs или репозиторий. Для статьи — arXiv/ACL/DOI, не сайт-саммари.
7. **Фильтр AI-контаминации.** «Retrieval Collapses When AI Pollutes the Web» (NAVER, WWW '26): **67% загрязнения пула даёт >80% загрязнения выдачи**, при этом точность ответов остаётся ~68–70% — «обманчиво здоровое» состояние, где проседает провенанс; BM25 пропускает ~19% adversarial контента в top-10 **[strong]**. Рекомендация авторов: defensive ranking по relevance + factuality + **provenance**, perplexity-фильтры, provenance-графы. Ahrefs (апрель 2025): 74.2% из 900k новых англоязычных страниц содержат AI-контент **[suggestive]**.
   Практично: дедуп почти-идентичного контента между доменами (SimHash/MinHash) — AI-фермы реплицируют текст, без дедупа «пять источников» окажутся одним **[speculative, но дёшево]**.
8. **Базовый уровень для сравнения.** Tow Center / CJR (март 2025), 8 AI-поисковиков × 200 запросов: >60% ответов с неверными атрибуциями; Perplexity 37% неверных, Grok 3 — 94%; **платный Perplexity Pro дал больше неверных цитат, чем бесплатный**. 0% фабрикации — не «базовая гигиена», а результат явного инженерного контроля.

## (d) Анти-фабрикация и анти-хеджирование

### Анти-фабрикация

| Контроль | Тип | Почему работает |
|---|---|---|
| Zero-query reject | детерминированный | уже есть |
| **URL ∈ transcript результатов** | детерминированный | фабрикация невозможна по построению, precision 100% |
| **Quote ⊂ transcript** | детерминированный | ловит пересказ, выдаваемый за цитату |
| urlhealth 4-класса + Wayback | сетевой | отделяет stale от hallucinated и от BLOCKED |
| Crossref для DOI | сетевой | DOI — худшее по точности поле в фабрикованных ссылках **[suggestive]** |
| Конъюнкция сигналов перед FABRICATED | правило | один промах БД → ~5× ложных срабатываний **[strong]** |
| Runtime-assert на backend | код | assert, что backend поддерживает search, до старта роли |

**Про базовые ставки.** HALLMARK failure mode (ii): при реалистичной базовой ставке ~2% решает **FPR, а не recall** — лучшие верификаторы дают 1 истинную галлюцинацию на 6–9 алертов, худшие 1 на 35+ **[strong]**. Если `fabricated_url_rate` стремится к нулю, агрессивный детектор будет генерировать почти только шум. Держать детектор консервативным, BLOCKED/UNKNOWN — отдельной строкой.

### Анти-хеджирование

1. **Не оптимизировать hedging_rate напрямую — это Goodhart.** Загнать долю `unverifiable` вниз тривиально: модель начнёт называть слабо подтверждённое `confirmed`. Правильная рамка — **selective prediction / risk-coverage**: coverage = доля решительных вердиктов, risk = ошибочность решительных, измеренная Stage 1–4. Метрики — risk-coverage curve, ECE/MCE **[suggestive]**. Алерт только когда coverage падает **без** роста precision.

2. **Главный рычаг: выводить вердикт механически из таблицы доказательств, а не из прозы модели.**
```
supported_domains ≥ 2 и contradicting = 0        → confirmed
contradicting ≥ 1 и supported = 0                → refuted
supported ≥ 1 и contradicting ≥ 1                → partial (conflicting)
supported = 1, независимого подтверждения нет     → partial (weak)
ни одной цитаты, прошедшей Stage 1–3             → unverifiable
```
Тогда hedging_rate — функция собранных доказательств, а не темперамента модели. **Единственная правка, решающая проблему в корне.**

3. **Сделать хеджирование дорогим, а не запрещённым.** `unverifiable` обязан нести обоснование: список запросов, число просмотренных результатов, число доменов, причина недостаточности. Требовать `queries ≥ N` и `results_seen ≥ M` до того, как метка допустима. Повышает цену отговорки, не поощряя ложную уверенность **[suggestive]**.

4. **Разбить `partial` на подтипы:** `partial-scope` (верно для подмножества), `partial-temporal` (устарело), `partial-conflicting` (источники расходятся). Каждый требует назвать различающее доказательство.

5. **Числовая confidence + калибровка.** ECE/Brier против исхода code-side pipeline на held-out наборе. Verbalized confidence у LLM систематически переуверен **[suggestive]**.

6. **Не добавлять LLM-судью для оценки хеджирования.** NAACL 2025: LLM-судьи не робастны к epistemic markers и имеют **негативный биас против выражений неуверенности** **[strong]**. Будет двойной счёт.

## (e) Ранжированные рекомендации

| # | Рекомендация | Impact | Effort | Risk |
|---|---|---|---|---|
| 1 | **Transcript allow-list для URL + quote-substring check** (offline) | Очень высокий | Низкий | Низкий |
| 2 | **Механический вывод вердикта из evidence-таблицы** | Очень высокий | Средний | Средний — калибровать пороги |
| 3 | **urlhealth: HEAD + Wayback + 4 класса, с кэшем** | Высокий | Низкий-средний | Низкий |
| 4 | **Query diversity gate + обязательный disconfirming query** | Высокий | Низкий | Низкий |
| 5 | **Независимые registrable domains вместо URL** | Высокий | Низкий | Низкий |
| 6 | **risk-coverage вместо голого hedging_rate** | Средний-высокий | Низкий | Низкий |
| 7 | **Подтипы `partial` / обоснование `unverifiable`** | Средний-высокий | Низкий | Низкий |
| 8 | **Локальный entailment: HHEM-2.1-open (0.1B, Apache 2.0)** | Средний-высокий | Средний | Средний — потолок ~75% |
| 9 | **Authority tiers + recency** | Средний | Средний | Средний |
| 10 | **Strict mode: MiniCheck-Flan-T5-L + ALCE leave-one-out** | Средний | Средний-высокий | Средний — 770M, латентность |
| 11 | **Adaptive stopping по domain-saturation** | Средний | Средний | Средний |
| 12 | **Crossref/DOI-верификация** | Низкий-средний | Низкий | Низкий |
| 13 | **SimHash-дедуп против AI-slop реплик** | Низкий-средний | Средний | Средний **[speculative]** |
| 14 | **Численная confidence + ECE на held-out** | Средний | Высокий — нужен golden set | Низкий |

**Стек:** `httpx` + `tenacity`, `trafilatura`, `pypdf`, `rapidfuzz`, `tldextract`, `rank_bm25`, `transformers`+`torch` (HHEM/MiniCheck), `crossrefapi`/`habanero`, `diskcache`/sqlite. **Пункты 1–7 не требуют ни ML, ни torch** — отдельный extras, ядро остаётся лёгким.

**Порядок:** 1 → 3 → 4/5 → 2 → 6/7 → 8.

## (f) Источники
- ALCE — https://arxiv.org/pdf/2305.14627
- Measuring Attribution in NLG (AIS) — https://arxiv.org/abs/2112.12870
- FActScore — https://arxiv.org/abs/2305.14251 · OpenFActScore — https://arxiv.org/pdf/2507.05965
- SAFE / LongFact — https://arxiv.org/pdf/2403.18802 · https://github.com/google-deepmind/long-form-factuality
- AttrScore — https://arxiv.org/pdf/2305.06311 · AttributionBench — https://arxiv.org/abs/2402.15089
- MiniCheck — https://arxiv.org/abs/2404.10774 · https://llm-aggrefact.github.io/
- HHEM-2.1-open — https://huggingface.co/vectara/hallucination_evaluation_model
- RAGAS metrics — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- DeepResearch Bench — https://arxiv.org/abs/2506.11763
- Claimify (ACL 2025) — https://arxiv.org/abs/2502.10855
- urlhealth / Reference Hallucinations — https://arxiv.org/abs/2604.03173
- HALLMARK — https://arxiv.org/html/2607.18360
- Tow Center / CJR — https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php
- Crossref REST API — https://www.crossref.org/documentation/retrieve-metadata/rest-api/
- Trafilatura evaluation — https://trafilatura.readthedocs.io/en/latest/evaluation.html
- Retrieval Collapses When AI Pollutes the Web (WWW '26) — https://arxiv.org/html/2602.16136v1
- Know Your Limits: Survey of Abstention (TACL) — https://arxiv.org/html/2407.18418
- Are LLM-Judges Robust to Expressions of Uncertainty? (NAACL 2025) — https://aclanthology.org/2025.naacl-long.452/
- TASR — https://arxiv.org/html/2606.13814
- GPT-Researcher — https://github.com/assafelovic/gpt-researcher · STORM — https://github.com/stanford-oval/storm
