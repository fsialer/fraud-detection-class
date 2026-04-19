# Santandors — Fraud Detection Class

A real-world Python engineering class built around a credit application fraud detection system. Students work through progressive issues to transform a messy data science notebook into production-grade software.

## The scenario

Santandors is a Django app where people apply for credit. A fraud pipeline should analyze their transaction history and determine whether applications get approved or rejected. Right now the fraud logic lives in a messy Jupyter notebook and everything runs against the production database — your job is to productionize it.

See [Issue #1](https://github.com/haro-ca/fraud-detection-class/issues/1) for the full epic, architecture diagrams (as-is vs to-be), and sub-issues.

## Setup

```bash
# clone and create your branch
git clone https://github.com/haro-ca/fraud-detection-class.git
cd fraud-detection-class
git checkout -b new-applications-approval<YOUR_TEAM_NUMBER>

# install dependencies
uv sync

# configure environment
cp .env.example .env
# replace PASSWORD in .env with the credential shared by the instructor

# run the app
uv run python manage.py runserver
```

## Project structure

```
├── config/                 # Django settings, urls, wsgi
├── creditapp/              # Django app (models, views, templates, admin)
├── src/
│   └── fraud/              # Extracted fraud rules (one file per rule)
├── scripts/
│   └── transactions.py     # Transaction feed simulator (instructor-only)
├── manage.py
└── pyproject.toml
```

## Issues

| # | Issue | Depends on |
|---|-------|------------|
| #1 | **Epic**: credit approvals must show in app after fraud pipeline | — |
| #2 | Extract fraud rules from notebook into `src/` modules | — |
| #9 | ↳ Extract each fraud rule into a callable function | — |
| #10 | ↳ Add tests for the extracted rules | #9 |
| #11 | ↳ Create entry point and wire notebook to extracted modules | #9, #10 |
| #3 | Batch ETL pipeline — Neon → DuckDB / Databricks (optional) | #2 |
| #4 | Fraud status API endpoint + Django integration | #3 |
| #5 | CLI for pipeline operations via Rich/Typer | #3 |
| #6 | Fraud review queue in Django admin | #4 |
| #7 | MCP server for application status | #4, #5 |
| #8 | Streamlit analytics dashboard over OLAP layer | #3 |
| #12 | Add logging to the fraud pipeline | #3 |

## Database permissions

Students have scoped access to the shared Neon database:

| Table | SELECT | INSERT | UPDATE | DELETE |
|---|---|---|---|---|
| `credit_applications` | yes | yes (via `/apply`) | `status` column only | no |
| `transactions` | yes | no | no | no |
| `fraud_results` | yes | yes | no | yes |

`scripts/transactions.py` is instructor-only — it generates live transaction data and new credit applications. You'll see data flowing in when the instructor runs it.

## Tech stack

- Python 3.12+
- Django 5.1+
- PostgreSQL (Neon) — OLTP
- DuckDB / Databricks — OLAP (you'll build this)
- Polars (not pandas)
- uv for dependency management
