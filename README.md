# GnuCash Enrich (Skeleton)

Local-first Python tooling to enrich unresolved GnuCash transactions using email and receipt evidence, with ML-assisted suggestions and mandatory user approval.

> Status: **Architecture + skeleton only**. Core adapters and engines currently expose interfaces/stubs and are not fully implemented yet.

## What this project is for

The target workflow is:

1. Read a GnuCash book (XML `.gnucash` or SQLite).
2. Find unresolved transactions (for example categories like `Unspecified` and `Imbalance-GBP`).
3. Gather evidence from:
   - `.eml` emails
   - receipt images (`jpg`, `jpeg`, `heic`)
   - historical categorized transactions
4. Generate suggested descriptions/categories/splits with confidence.
5. Review **one transaction at a time** in a local web app.
6. Apply only explicitly approved changes, with backup + audit + rollback journal.

The full product specification is captured in:
- `gnucash_email_receipt_categorization_spec.mdc`
- `architecture_implementation_plan.mdc`

---

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── gnucash_email_receipt_categorization_spec.mdc
├── architecture_implementation_plan.mdc
└── src/gnc_enrich/
    ├── __main__.py
    ├── cli.py
    ├── config.py
    ├── domain/models.py
    ├── gnucash/loader.py
    ├── email/{parser.py,index.py}
    ├── receipt/{ocr.py,repository.py}
    ├── matching/{email_matcher.py,receipt_matcher.py}
    ├── ml/predictor.py
    ├── review/{service.py,webapp.py}
    ├── state/repository.py
    ├── apply/engine.py
    └── services/pipeline.py
```

---

## Prerequisites

- Python **3.11+**
- Local filesystem access to:
  - GnuCash file
  - directory of `.eml` files
  - directory of receipt images

Optional (future implementation):
- OCR engine dependencies
- ML/LLM dependencies
- web UI framework dependencies

---

## Installation / setup

### Option A: run from source (recommended right now)

```bash
cd /path/to/GnuCashUtils
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Option B: module execution with `PYTHONPATH`

If not installing editable package:

```bash
cd /path/to/GnuCashUtils
PYTHONPATH=src python -m gnc_enrich --help
```

---

## CLI commands (current skeleton)

The project exposes three top-level commands:

```bash
gnc-enrich run ...
gnc-enrich review ...
gnc-enrich apply ...
```

Equivalent module form:

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
  --amount-tolerance 0.50
```

Arguments:

- `--gnucash-path` (required): XML or SQLite GnuCash file.
- `--emails-dir` (required): `.eml` folder.
- `--receipts-dir` (required): `jpg|jpeg|heic` folder.
- `--processed-receipts-dir` (required): destination for approved/matched receipts.
- `--state-dir` (required): index/proposal/audit state folder.
- `--date-window-days` (optional, default `7`): evidence date window.
- `--amount-tolerance` (optional, default `0.50`): amount match tolerance.
- `--include-skipped` (flag): include previously skipped tx in candidate set.

> Current behavior: parses CLI config successfully; business pipeline implementation is pending.

### 2) `review` — launch local review application

```bash
python -m gnc_enrich review \
  --state-dir /data/gnc-enrich-state \
  --host 127.0.0.1 \
  --port 7860
```

Arguments:
- `--state-dir` (required): persisted proposals and evidence state.
- `--host` (default `127.0.0.1`)
- `--port` (default `7860`)

> Current behavior: CLI contract exists; web app runtime implementation is pending.

### 3) `apply` — write approved decisions

```bash
python -m gnc_enrich apply \
  --state-dir /data/gnc-enrich-state \
  --create-backup \
  --backup-dir /data/backups \
  --in-place
```

Arguments:
- `--state-dir` (required): proposal/decision/audit state.
- `--create-backup` (flag): request backup snapshot before write.
- `--backup-dir` (optional): backup destination.
- `--in-place` (flag): apply changes directly to source GnuCash file.

> Current behavior: CLI contract exists; apply engine implementation is pending.

---

## Example end-to-end session (target workflow)

```bash
# 1) Build candidate proposals from data sources
python -m gnc_enrich run \
  --gnucash-path /finance/books.gnucash \
  --emails-dir /finance/emails \
  --receipts-dir /finance/receipts/new \
  --processed-receipts-dir /finance/receipts/processed \
  --state-dir /finance/gnc-state \
  --date-window-days 7 \
  --amount-tolerance 0.50

# 2) Review one-by-one in local web app
python -m gnc_enrich review --state-dir /finance/gnc-state --host 127.0.0.1 --port 7860

# 3) Apply only approved decisions with backup
python -m gnc_enrich apply --state-dir /finance/gnc-state --create-backup --backup-dir /finance/backups --in-place
```

---

## Data/state conventions (planned)

Expected state artifacts include:

- `email_index.*`
- `receipt_index.*`
- `candidate_transactions.*`
- `proposals.*`
- `skip_state.*`
- `feedback_events.*`
- `audit_log.jsonl`
- `apply_journal.*`

These are defined in the spec and architecture docs; concrete persistence formats are pending implementation.

---

## Development notes

### Quick validation commands

```bash
# Show CLI contract
PYTHONPATH=src python -m gnc_enrich --help

# Python syntax sanity check
python -m compileall src
```

### Current implementation status

Implemented now:
- package structure
- domain/config models
- module/class contracts
- CLI argument interface

Pending implementation:
- GnuCash XML/SQLite adapters
- email parsing/indexing
- receipt OCR + matching
- ML predictor and feedback loop
- one-by-one review web app
- apply/rollback/audit engine

---

## Safety and usage guidance

- Do **not** assume write/apply behavior exists yet beyond CLI argument handling.
- Treat current code as scaffold for multi-agent implementation.
- Use backups for any future apply flow that edits financial records.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'gnc_enrich'`

Use one of:

```bash
pip install -e .
```

or

```bash
PYTHONPATH=src python -m gnc_enrich --help
```

### Command exits immediately without doing processing

This is expected in the current skeleton; core service implementations currently raise `NotImplementedError` or are not wired yet.

---

## License

No license file has been added yet.
