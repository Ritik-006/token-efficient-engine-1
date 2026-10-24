# PromptForge AI — Core Pipeline + IntentLock Validation

A Streamlit application that analyzes prompts, creates ContextMax and TokenLite candidates, extracts machine-checkable requirements, validates candidates, blocks unsafe recommendations, and estimates input-token costs.

## Features
- End-to-end input → analysis → optimization → requirement ledger → validation → recommendation pipeline.
- Conservative validation for exact numeric counts, numeric limits, literal values, quoted values, prohibitions, required inclusions, JSON key names, and output-format mentions.
- Editable IntentLock ledger: users can add/remove requirements, edit requirement text and toggle criticality, then revalidate.
- Fail-closed recommendation gate: critical checks must pass and estimated input-token savings must be at least 10%.
- Word-level original-vs-TokenLite diff.
- Transparent per-call, daily and monthly input-cost estimates with user-entered price source/date.
- TXT and structured JSON report exports.
- 8,000-token MVP input limit with a clear error message.
- Safe and intentional-failure demo examples.
- Automated regression tests for empty prompts, changed numbers, missing limits, removed prohibitions, missing JSON keys, and cost calculations.
- No LLM API key required; the current implementation is deterministic and runs locally.

## Run on Windows (PowerShell)
```powershell
cd PromptForge_AI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run tests
From the project folder:
```powershell
python -m unittest -v
```
or:
```powershell
python tests.py
```
Tests use Python's built-in `unittest` and do not require a running Streamlit server.

## Demo plan
1. Start the app and select **Safe optimization example**.
2. Click **Analyze, optimize & validate**.
3. Show extracted requirements, candidate status, token metrics and cost estimates.
4. Run the tests to show a deliberately altered candidate fails when it loses a word limit, changes a number, drops a prohibition, or omits a JSON key.
5. Explain that the validator is intentionally conservative and that PASS is not a guarantee of semantic equivalence.

## Validation design and limitations
- Requirements are extracted with deterministic regular expressions. Natural-language interpretation is incomplete; always review the ledger.
- Checks compare extracted clauses/literals conservatively. A candidate can pass lexical checks yet still alter meaning or produce a different model response.
- When no machine-checkable requirements are found, the status is `UNCERTAIN`, not `PASS`.
- Token counts use `tiktoken` `cl100k_base` when available. Otherwise a rough estimate is used. This is not guaranteed to match every model's tokenizer.
- Cost estimates are for input tokens only and depend on the price entered by the user. Verify the current provider price before using the estimate.
- The optimizer does not call an LLM. It uses conservative deterministic transformations; no external API key is needed.
- A true answer-quality evaluation requires running original and optimized prompts through the same model and evaluating their outputs against a fixed rubric.

## Project files
- `app.py`: Streamlit dashboard.
- `optimizer.py`: analysis, extraction, optimization, validation and estimates.
- `tests.py`: regression tests.
- `requirements.txt`: Python dependencies.

## Repeatable benchmark

Run `python benchmark.py` from the project directory. It processes 10 fixed prompt cases and writes `benchmark_results.csv` with per-case token estimates, validation outcomes, and local pipeline latency. The printed check pass rate is a syntactic/lexical check rate, **not** a semantic quality score. Record the actual output from your machine; do not present target metrics as achieved results.
