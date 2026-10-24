1. “About” box (short description)
Fail-closed prompt optimizer: generates two prompt rewrites, validates every critical requirement with an editable IntentLock ledger, and only recommends savings that are safe. Offline, deterministic, Streamlit + tiktoken.

This is about 200 characters, under GitHub’s 350 limit.

2. Topics (tags)
prompt-engineering, prompt-optimization, token-efficiency, llm, llm-cost, tiktoken, streamlit, python, ai-tools, requirement-validation
3. README intro (paste at the top of README.md)
markdown
# ⚡ PromptForge AI
### Prompt Optimization & Token Efficiency Engine

> **Shorter is only better when the requirements survive.**

PromptForge AI is a Streamlit app that reduces the input-token cost of LLM prompts
without silently dropping what matters. Most prompt shorteners cut words and hope
for the best. PromptForge treats optimization as a constrained problem: **minimise
tokens, subject to every critical requirement passing.**

## How it works
1. **Extract:** a regex engine builds an editable **IntentLock ledger** of requirements
   (exact counts, limits, numbers, quoted text, prohibitions, JSON keys, output formats).
2. **Generate:** two candidates are produced.
   - **ContextMax:** keeps the full original plus execution guidance, for precision.
   - **TokenLite:** removes low-risk filler, for lower cost.
3. **Validate and gate:** every candidate is checked against every ledger item.
   A critical FAIL or UNCERTAIN blocks it. A candidate is recommended only if all
   critical checks pass **and** estimated input-token savings are at least 10%.

## Features
- Editable requirement ledger with criticality toggles and live re-validation
- Word-level diff of original vs. TokenLite
- Token counts via `tiktoken` (`cl100k_base`), with an offline fallback estimate
- Per-call, daily and monthly input-cost estimates (user-entered price, source and date)
- TXT and JSON report export
- 8,000-token input limit with a clear error message
- Built-in safe and intentional-failure demo prompts
- Fully local: no API key, no prompt storage

## Project structure
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `optimizer.py` | Analysis, requirement extraction, optimization, validation, cost |
| `tests.py` | 12 regression tests (`unittest`) |
| `benchmark.py` | Repeatable 10-case benchmark, writes `benchmark_results.csv` |

## Quick start
​```bash
pip install -r requirements.txt
streamlit run app.py
python tests.py          # run tests
python benchmark.py      # run benchmark
​```

## Measured results (local run, 10 benchmark cases)
- 12/12 regression tests pass
- 82/82 extracted lexical checks pass (not a semantic-quality score)
- Median pipeline latency: 0.23 ms (P95: 2.5 ms)
- Median TokenLite savings: 5.5%. This is below our 20% target and is being improved

## Limitations
Validation is lexical and conservative, so a PASS does not prove semantic equivalence.
Regex extraction can miss implicit constraints, so always review the ledger. Costs cover
input tokens only. A/B output-quality testing is planned.

## Roadmap
Higher TokenLite savings, spaCy-based objective/style checks, LLM adapter with A/B
quality testing, and a 50-prompt annotated benchmark.

## Team Anonymous
Prashanth Kutre (Lead) · Ritik Kamble · Owais Watangi · Adati Pawar

When you paste this, remove the invisible character before the closing code-fence lines in the Quick start block. I added it only so the example would display correctly here.

Before you push
Fix the crash in app.py. It uses date.today() but never imports date. Add from datetime import date at the top, or the sidebar will fail.
Add a .gitignore entry for __pycache__/ and .venv/, if your .gitignore doesn’t already have them.
Add a LICENSE (MIT is common for hackathon projects) so others know they can reuse it.
