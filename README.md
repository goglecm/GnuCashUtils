## GnuCash Enrich

Local-first Python tooling that helps you resolve unresolved GnuCash transactions using email and receipt evidence, with ML-assisted suggestions and mandatory user approval.

> Status: **v1 implemented** — behaviour and acceptance criteria are defined in the main specification.

---

### Overview

- **Input**: A GnuCash book (gzip-compressed XML `.gnucash`), a directory of `.eml` emails, and a directory of receipt images.
- **Goal**: Find transactions with splits to `Unspecified` / `Imbalance-GBP`, suggest better descriptions and categories, and let you approve or edit each one.
- **How it works**:
  - Reads the GnuCash file and identifies candidate transactions.
  - Indexes emails and OCRs receipts.
  - Uses ML (and optionally an LLM) to suggest descriptions/categories.
  - Presents proposals **one transaction at a time** in a local web app.
  - Applies only the changes you explicitly approve, with backups and rollback.

For full domain rules, evidence rules, and LLM behaviour, see the **Specification** linked below.

---

### Prerequisites

- Python **3.11+**
- **Tesseract OCR** (for receipt images):
  - Fedora: `sudo dnf install tesseract`
  - Ubuntu: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`

---

### Installation

```bash
cd /path/to/GnuCashUtils
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

After installation, the `gnc-enrich` console script is available (equivalent to `python -m gnc_enrich`).

---

### Core commands (user guide)

Global flag: `-v` / `--verbose` enables DEBUG-level logging for all subcommands. **Note:** DEBUG logs may include transaction descriptions, email snippets, and LLM request/response content; avoid in shared or untrusted environments.

#### 1) `run` — build proposals from source data

```bash
python -m gnc_enrich run \
  --gnucash-path /data/books/main.gnucash \
  --emails-dir /data/mail/eml \
  --receipts-dir /data/receipts/incoming \
  --processed-receipts-dir /data/receipts/processed \
  --state-dir /data/gnc-enrich-state
```

Key options:

- **`--gnucash-path`**: Gzip-compressed XML GnuCash file.
- **`--emails-dir`**: Directory of `.eml` files (scanned recursively).
- **`--receipts-dir`** / **`--processed-receipts-dir`**: Receipt images in, processed receipts out.
- **`--state-dir`**: Directory for indexes, proposals, decisions, and audit. **May contain secrets**: if you use LLM API keys, they are stored in plain text in `run_config.json`. Restrict filesystem access to this directory.
- **`--date-window-days`**, **`--amount-tolerance`**: Evidence matching controls.
- **`--include-skipped`**: Re-process transactions previously skipped in review.
- **LLM flags**: `--llm-mode`, `--llm-endpoint`, `--llm-model`, `--llm-use-web`, `--llm-warmup-on-start`, `--llm-timeout`, `--llm-extraction-endpoint`, `--llm-extraction-model` (see spec for detailed behaviour).

#### 2) `review` — launch local review application

```bash
python -m gnc_enrich review \
  --state-dir /data/gnc-enrich-state \
  --host 127.0.0.1 \
  --port 7860
```

Opens a Flask web app at `http://127.0.0.1:7860` for one-by-one review with:

- Queue view with **To categorise** and **Approved transactions** sections.
- A transaction page showing the original transaction, ML suggestion, optional LLM suggestion, email and receipt evidence, and a decision form (approve / skip, with description and splits editable).

#### 3) `apply` — write approved decisions

```bash
# Preview changes only (no write)
python -m gnc_enrich apply --state-dir /data/gnc-enrich-state --dry-run

# Apply with backups written to a chosen directory
python -m gnc_enrich apply \
  --state-dir /data/gnc-enrich-state \
  --create-backup \
  --backup-dir /data/backups \
  --in-place
```

Key options:

- **`--dry-run`**: Generate a human-readable report and exit without writing.
- **`--create-backup` / `--no-backup`**: Control backup creation (default: create).
- **`--backup-dir`**: Where backups are written (default: `<state-dir>/backups`).
- **`--in-place` / `--no-in-place`**: Modify the original GnuCash file or write a new one.
- **`--backup-retention N`**: Keep at most N backups per book (default: unlimited).

#### 4) `rollback` — restore from backup

```bash
# List available backups for a state directory
python -m gnc_enrich rollback --state-dir /data/gnc-enrich-state --list-backups

# Restore the most recent backup
python -m gnc_enrich rollback --state-dir /data/gnc-enrich-state

# Restore a specific backup file by name
python -m gnc_enrich rollback \
  --state-dir /data/gnc-enrich-state \
  --backup books.20250101T120000Z.gnucash
```

If the state directory has never been through `run` (and thus has no `run_config.json`), `rollback` prints a clear error and exits with a non-zero code.

---

### Example end-to-end session

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
python -m gnc_enrich apply \
  --state-dir /finance/gnc-state \
  --create-backup \
  --backup-dir /finance/backups \
  --in-place
```

---

### LLM usage (high level)

LLM integration is **optional** and **disabled by default**. At a high level:

- During **run**, proposals are ML-only unless advanced configuration enables LLM use for low-confidence cases.
- During **review**, the **“Check with LLM”** button can:
  - Enrich terse email or receipt information (seller, items, order IDs).
  - Suggest an improved description and category.
- Receipt OCR can optionally fall back to an LLM when Tesseract fails to extract a total.

For the full LLM interaction model (extraction vs category, cross-product of modes, and prompt templates), see the specification’s LLM section.

### Security and privacy

- **State directory**: Stores proposals, decisions, and run metadata. When LLM is configured with API keys, they are persisted in `run_config.json` in plain text. Restrict read/write access to the state directory (e.g. `chmod 700`).
- **Logging**: At INFO level, only sizes and timings are logged. With `-v` (DEBUG), logs may contain transaction text, email content, and LLM inputs/outputs; do not enable DEBUG on shared systems or when log output is not trusted.
- **Review app**: Intended for local use (`127.0.0.1`). Do not expose the review server to untrusted networks without authentication.

---

### Repository layout (for orientation)

```text
src/gnc_enrich/
  cli.py               # CLI: run / review / apply / rollback
  config.py            # RunConfig, ApplyConfig, ReviewConfig, LlmConfig
  domain/models.py     # Transactions, splits, evidence, proposals, decisions, audit
  gnucash/loader.py    # GnuCash XML loader/writer
  email/               # Email parser + index
  receipt/             # Receipt OCR + repository
  matching/            # Email and receipt matching
  ml/                  # CategoryPredictor + FeedbackTrainer
  review/              # Review service + Flask web app
  apply/               # ApplyEngine (dry-run, apply, backup, rollback, audit)
  state/               # JSON/JSONL state repository
  services/pipeline.py # Run-phase orchestration
docs/
  gnucash_email_receipt_categorization_spec.mdc
  developers_guide.mdc
  test_plan_and_strategy.mdc
```

---

### Documentation map (single source of truth)

- **Specification**: `docs/gnucash_email_receipt_categorization_spec.mdc`  
  Full description of domain rules, inputs/outputs, components, evidence matching, LLM flows, and acceptance criteria.

- **Developers guide**: `docs/developers_guide.mdc`  
  Development environment, codebase structure, documentation and testing conventions, and how to keep the spec and tests in sync.

- **Test plan & strategy**: `docs/test_plan_and_strategy.mdc`  
  Test levels, test matrix, spec coverage mapping, and future testing enhancements.

The README is intentionally an overview and user guide; when in doubt about behaviour, treat the **specification** as the source of truth.

---

### Running tests

```bash
pytest              # all tests
pytest -k matching  # subset by keyword
pytest tests/test_integration.py -v  # integration only
```

For full testing expectations and the current test matrix, see the developers guide and test plan.

---

### Licence

This project is licensed under the GNU General Public License v3.0. See `LICENSE.md` for details.
