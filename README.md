# Kaidzen

Polishes a raw idea with facts from web search — and evolves the very instructions that drive the polishing.

Two levels:

- **Level 1** takes an idea of 1–3 paragraphs and runs it through an Analyzer → Researcher → Refiner → Judge loop. The output is a polished idea, an assumption table with verdicts and source links, rubric scores, and a list of things that can only be validated by experiment.
- **Level 2** evolves the instructions of Level 1: it diagnoses weaknesses, generates mutations, runs them on a benchmark of ideas, compares results blindly, and promotes the winner only if objective metrics don't regress.

Runs **on a Claude subscription, no API key required**. OpenAI, DeepSeek, and Anthropic keys can be plugged in optionally, per role.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

No keys needed: the default backend is `subscription` — it drives the installed `claude` CLI and runs off your subscription.

Polish an idea:

```bash
python -m kaidzen run my-idea.md --domain business
```

A run takes several minutes; progress is printed as it goes. The result is `runs/<date>-<idea>/report.md`.

An interrupted run resumes from where it stopped, and the report can be rebuilt without calling the model:

```bash
python -m kaidzen resume runs/<id>/
python -m kaidzen report runs/<id>/
```

---

## How Level 1 works

| Role | What it does |
|---|---|
| **Analyzer** | Breaks the idea down into problem, audience, mechanics, and an **assumption registry** — everything the idea takes on faith. Assigns a criticality to each assumption. |
| **Researcher** | Takes the riskiest unverified assumptions and checks them via web search. Returns facts with links and a verdict: confirmed, refuted, partial, or unverifiable by search. |
| **Refiner** | Rewrites the idea based on the findings and the judge's critique. Every edit must reference a closed assumption or a critique item. |
| **Judge** | Scores the new version against a five-axis rubric, compares it to the previous one, and writes critique for the next iteration. |

The loop stops when the score gain plateaus, the iteration limit is exhausted, or all critical assumptions are closed.

### What guards against "polishing in a vacuum"

The main danger of a system like this: the text gets prettier while the idea doesn't get better. Guards are built in at three levels and verified by live runs:

- The Refiner can't make an edit without referencing a fact or a critique item — the orchestrator rejects such a response and asks again.
- The Judge doesn't see the Refiner's changelog: it scores the result, not the story about the work.
- A search-enabled call that made zero queries doesn't count. Otherwise the model happily produces a plausible but fabricated URL — this was observed in practice.

---

## How Level 2 works

```bash
python -m kaidzen evolve --domain business --generations 3
python -m kaidzen checkpoint evolve/<id>/            # read the summary
python -m kaidzen checkpoint evolve/<id>/ --approve  # or --reject
python -m kaidzen evolve-stop evolve/<id>/           # graceful stop
python -m kaidzen evolve-resume evolve/<id>/
```

One generation: diagnose the champion's weaknesses → two mutations → run each on the train ideas → blind pairwise comparison of reports → Gate → possible champion change.

`evolve` never starts on its own: no daemon, no schedule. A graceful stop finishes the current generation so already-completed runs aren't thrown away.

### Three defenses against Goodhart

The system optimizes metrics, and metrics can be gamed. Each defense came not from theory but from the evolution actually trying to cheat:

1. **Objective metrics are computed by code, not by a model.** Share of closed assumptions, share of hedging verdicts, edit groundedness, token spend. A pretty report with worse numbers doesn't get promoted.
2. **Absolute counts next to ratios.** The very first evolution pushed the closed-assumption ratio to 100% by simply dropping questions from the registry — same numerator, smaller denominator. Now shrinking the registry by more than 20% requires the number of closed assumptions to grow **in absolute terms**.
3. **Blind comparison and holdout.** The judge sees only two anonymized reports and compares each pair twice with positions swapped — a disagreement counts as a tie. Some ideas don't participate in evolution and are checked at checkpoints: better on training ideas but not on held-out ones means overfitting.

The meta level **does not evolve itself**: the diagnostician, mutator, and judge prompts are edited by hand. If the system started improving its own judge, the ruler would drift along with what it measures — details in `docs/specs/2026-08-03-evolution-memory.md` §5.

### Memory between runs

`candidates/EVOLUTION-<domain>.json` is a log of all mutation attempts with their outcomes. The diagnostician sees what has already been tried and what was rejected; the mutator gets the list of accepted findings marked "do not break". Without the log, every run started from scratch and repeated past mistakes.

---

## Candidates, backends, models

A **candidate** is a swappable instruction set: `config.yaml` plus prompts for the five roles. The domains `generic`, `business`, and `games` are just different candidates. The current champion of a domain is recorded in `candidates/CHAMPION-<domain>`.

Model and transport are configured **per role**:

```yaml
backends:
  subscription: { type: claude_agent_sdk }
  deepseek:     { type: openai_compat, base_url: "https://api.deepseek.com", api_key_env: DEEPSEEK_API_KEY }
roles:
  researcher: { backend: subscription, model: claude-sonnet-5 }
  judge:      { backend: subscription, model: claude-opus-5 }
```

Web search is supported by `subscription` and `anthropic`. `deepseek` and `openai` don't have it, so the config **won't let** you assign the Researcher to them — without search it can't close assumptions with facts. Any other role can go there.

Keys are read from environment variables or from `.env` (see `.env.example`). The config stores only the variable name, never the value itself.

---

## Layout

```
kaidzen/            code: roles, orchestrators, backends, metrics, Gate
candidates/         candidates (instruction sets) + champion pointers + evolution log
benchmark/          ideas for evolution: <domain>/ideas/*.md
runs/               Level 1 runs        (gitignored)
evolve/             Level 2 runs        (gitignored)
docs/specs/         specs
docs/superpowers/   implementation plans
scripts/            one-off utility scripts
tests/              501 tests
```

Note: the role prompts, specs, and benchmark ideas are written in Russian — that's the language the system has been developed and evaluated in.

## Development

```bash
.venv/bin/pytest -q
.venv/bin/pytest --cov=kaidzen -q
```

Tests don't touch the network. Live runs, however, kept finding what tests couldn't see — a broken retry protocol, a malformed model request, name collisions between runs, a collapsed comparison sample. Before trusting a change to the orchestrator or a backend, run one live generation.

## License

[MIT](LICENSE)
