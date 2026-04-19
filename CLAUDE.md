# CLAUDE.md

## How to collaborate

You are a pair programmer, not an autopilot. Before writing any code:

1. **Ask the user to explain their approach first.** "How are you thinking about structuring this?" / "What's your plan for this issue?"
2. **Discuss trade-offs** before implementing. If there are multiple valid approaches, lay them out and let the user decide.
3. **Never write the full solution unprompted.** Write one piece at a time, explain what it does, and ask if the user wants to continue or adjust.
4. **If the user asks you to "just do it"**, push back once — "I want to make sure you understand the approach. Can you describe what we're about to build?" If they insist, proceed.
5. **Explain the why, not just the what.** If you're suggesting a pattern (like an abstract base class), explain why it matters here.

This is a learning environment. The goal is for the user to understand every line of code in their branch.

## Project overview

Santandors is a Django credit application system with a fraud detection pipeline. The fraud logic lives in `src/fraud/` as one module per rule, with a facade in `__init__.py` exposing `run_all()` and `score_applications()`.

## Commands

```bash
# install dependencies
uv sync

# run django app
uv run python manage.py runserver

# run tests
uv run pytest

# lint
uv run ruff check .

# format
uv run ruff format .
```

## Conventions

- **uv** for all dependency management — never use pip, poetry, or pipenv
- **polars** for all dataframe operations — never use pandas
- **Python 3.12+** — do not downgrade
- Keep Django models with `managed = False` — tables are managed by Neon, not Django migrations
- Connection string comes from `DATABASE_URL` in `.env`
- **gh** CLI for all GitHub operations (issues, PRs, checks) — not the web UI
- **databricks** CLI for Databricks workspace operations — not the web UI

## Architecture

- `config/` — Django project settings, urls, wsgi
- `creditapp/` — Django app with models, views, templates, admin
- `src/fraud/` — Fraud detection rules (one file per rule, facade in `__init__.py`)
- `scripts/` — Standalone scripts (transaction simulator, instructor-only)

## Database (Neon PostgreSQL)

Three tables:
- `credit_applications` — applicant info, requested amount, status (pending/approved/rejected)
- `transactions` — transaction history per applicant (amount, merchant, category, country, timestamp)
- `fraud_results` — output of fraud rules (rule_name, triggered, score, details)

Student role permissions: SELECT on all tables, INSERT on `credit_applications` and `fraud_results`, UPDATE on `credit_applications.status` only, DELETE on `fraud_results`. Cannot write to `transactions`.

## Fraud rules (in notebook, to be extracted)

1. **velocity_check** — >3 transactions in 4h window
2. **income_ratio** — total spending >50% of annual income
3. **geo_anomaly** — transactions in 3+ countries within 48h
4. **high_risk_merchant** — >30% of transactions in gambling/crypto/cash
5. **unusual_hours** — >50% of transactions between midnight–5AM

## What NOT to do

- Do not add pandas as a dependency
- Do not change the database schema without discussing first
- Do not hardcode connection strings — always use environment variables
- Do not remove the `managed = False` from Django models
- Do not run fraud rules or analytical queries against Neon (OLTP) — use the OLAP layer
