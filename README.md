# GnuCash Enrich

Local-first Python tooling to enrich unresolved GnuCash transactions using email and receipt evidence, with ML-assisted suggestions and mandatory user approval.

> Status: **v1 Implemented** — all modules complete, 142 tests passing.

## What this project does

1. Reads a GnuCash book (gzip-compressed XML `.gnucash`).
2. Finds unresolved transactions (categories like `Unspecified` and `Imbalance-GBP`).
3. Gathers evidence from:
   - `.eml` emails (parsed, indexed, searched by date/amount/text)
   - Receipt images (`jpg`, `jpeg`, `png`, `heic`, `heif`) via Tesseract OCR
   - Historical categorized transactions
4. Generates suggested descriptions/categories/splits with ML confidence scores.
5. Presents proposals **one transaction at a time** in a local Flask web app.
6. Applies only explicitly approved changes, with backup + audit + rollback journal.

The full specification is in:
- `gnucash_email_receipt_categorization_spec.mdc`
- `architecture_implementation_plan.mdc`

---

## Prerequisites

- Python **3.11+**
- **Tesseract OCR** (system package):
  - Fedora: `sudo dnf install tesseract`
  - Ubuntu: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`

---

## Installation

```bash
cd /path/to/GnuCashUtils
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

---

## CLI commands

The project exposes three top-level commands:

```bash
python -m gnc_enrich run ...
python -m gnc_enrich review ...
python -m gnc_enrich apply ...
```

### 1) `run` — build proposals from source data

```bash
python -m gnc_enrich run \
  --gnucash-path /data/books/main.gnucash \
  --emails-dir /data/mail/eml \
  --receipts-dir /data/receipts/incoming \
  --processed-receipts-dir /data/receipts/processed \
  --state-dir /data/gnc-enrich-state \
  --date-window-days 7 \
  --amount-tolerance 0.50 \
  --llm-mode disabled
```

Arguments:

- `--gnucash-path` (required): Gzip-compressed XML GnuCash file.
- `--emails-dir` (required): Directory of `.eml` files.
- `--receipts-dir` (required): Directory of receipt images (`jpg|jpeg|png|heic|heif`).
- `--processed-receipts-dir` (required): Destination for approved/matched receipts.
- `--state-dir` (required): Index/proposal/audit state directory.
- `--date-window-days` (default `7`): Evidence date window (±days).
- `--amount-tolerance` (default `0.50`): Amount match tolerance in GBP.
- `--include-skipped` (flag): Re-process previously skipped transactions.
- `--llm-mode` (`disabled`|`offline`|`online`): LLM integration mode.
- `--llm-endpoint`: LLM API endpoint URL.
- `--llm-model`: LLM model name.

### 2) `review` — launch local review application

```bash
python -m gnc_enrich review \
  --state-dir /data/gnc-enrich-state \
  --host 127.0.0.1 \
  --port 7860
```

Opens a Flask web app at `http://127.0.0.1:7860` for one-by-one transaction review with evidence display, approve/edit/skip controls.

### 3) `apply` — write approved decisions

```bash
# Dry-run (preview only)
python -m gnc_enrich apply --state-dir /data/gnc-enrich-state --dry-run

# Apply with backup
python -m gnc_enrich apply \
  --state-dir /data/gnc-enrich-state \
  --create-backup \
  --backup-dir /data/backups \
  --in-place
```

Arguments:
- `--state-dir` (required): Proposal/decision/audit state directory.
- `--dry-run` (flag): Generate a human-readable report without writing changes.
- `--create-backup` (flag): Create a timestamped backup before writing.
- `--backup-dir`: Backup destination directory.
- `--in-place` (flag): Apply changes directly to source GnuCash file.

---

## Example end-to-end session

```bash
# 1) Build candidate proposals from data sources
python -m gnc_enrich run \
  --gnucash-path /finance/books.gnucash \
  --emails-dir /finance/emails \
  --receipts-dir /finance/receipts/new \
  --processed-receipts-dir /finance/receipts/processed \
  --state-dir /finance/gnc-state

# 2) Review one-by-one in local web app
python -m gnc_enrich review --state-dir /finance/gnc-state

# 3) Preview changes
python -m gnc_enrich apply --state-dir /finance/gnc-state --dry-run

# 4) Apply only approved decisions with backup
python -m gnc_enrich apply --state-dir /finance/gnc-state --create-backup --backup-dir /finance/backups --in-place
```

---

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── gnucash_email_receipt_categorization_spec.mdc
├── architecture_implementation_plan.mdc
├── developers_guide.mdc
├── test_plan_and_strategy.mdc
├── src/gnc_enrich/
│   ├── __main__.py
│   ├── cli.py                       # CLI entry points (run/review/apply)
│   ├── config.py                    # Configuration dataclasses
│   ├── domain/models.py             # Domain entities
│   ├── gnucash/loader.py            # GnuCash XML parser and writer
│   ├── email/parser.py              # .eml file parser
│   ├── email/index.py               # Persistent email index (JSONL)
│   ├── receipt/ocr.py               # Tesseract OCR + optional LLM fallback
│   ├── receipt/repository.py        # Receipt file management
│   ├── matching/email_matcher.py    # Email-to-transaction scoring
│   ├── matching/receipt_matcher.py  # Receipt-to-transaction matching
│   ├── ml/predictor.py              # ML category prediction + feedback
│   ├── review/service.py            # Review queue management
│   ├── review/webapp.py             # Flask web app
│   ├── review/templates/            # Jinja2 HTML templates
│   ├── apply/engine.py              # Apply, backup, rollback, audit
│   ├── state/repository.py          # JSON/JSONL state persistence
│   └── services/pipeline.py         # Pipeline orchestration
└── tests/                           # 142 tests
    ├── conftest.py                  # Shared fixtures
    └── fixtures/emails/             # Sample .eml files
```

---

## State artifacts

The `--state-dir` directory contains:

| File | Format | Purpose |
|------|--------|---------|
| `email_index.jsonl` | JSONL | Parsed email evidence |
| `email_index_manifest.json` | JSON | Tracks indexed `.eml` filenames |
| `proposals.json` | JSON | Generated proposals with evidence |
| `decisions.jsonl` | JSONL | Review decisions (append-only) |
| `skip_state.json` | JSON | Skipped transaction records |
| `feedback_events.jsonl` | JSONL | User feedback for retraining |
| `audit_log.jsonl` | JSONL | Complete audit trail |
| `apply_journal.jsonl` | JSONL | Undo journal for rollback |
| `run_config.json` | JSON | Metadata from last pipeline run |

---

## Running tests

```bash
pytest              # all 142 tests
pytest -v           # verbose
pytest -k matching  # keyword filter
pytest tests/test_integration.py -v  # integration only
```

See `developers_guide.mdc` for full testing documentation.

---

## LLM configuration

The system supports optional LLM integration (disabled by default):

```bash
# Local LLM (e.g., Ollama)
python -m gnc_enrich run --llm-mode offline \
  --llm-endpoint http://localhost:11434/v1/chat/completions \
  --llm-model llama3 ...

# Remote LLM (e.g., OpenAI-compatible API)
python -m gnc_enrich run --llm-mode online \
  --llm-endpoint https://api.example.com/v1/chat/completions \
  --llm-model gpt-4 ...
```

LLM is used for:
1. **Receipt OCR fallback** when Tesseract fails to extract a total.
2. **Category rationale** when ML classifier confidence is below 60%.

---

## Troubleshooting

- **`ModuleNotFoundError`**: Run `pip install -e ".[dev]"`
- **Tesseract not found**: Install the system package (see Prerequisites)
- **Bytecode clutter**: `find . -type d -name '__pycache__' -prune -exec rm -rf {} +`

---

## License

No license file has been added yet.
