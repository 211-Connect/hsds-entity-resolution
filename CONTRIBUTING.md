# Contributing

Thanks for contributing to `hsds_entity_resolution`.

## Scope and intent

This project supports community organizations that share HSDS data by:

- deduplicating records across sources
- orchestrating continual quality checks
- enabling durable, long-running data sharing partnerships

## Pull request workflow

1. Create a branch from the latest main branch.
2. Keep changes focused and easy to review.
3. Add or update tests when behavior changes.
4. Run required local quality checks (below).
5. Open a PR with:
   - what changed
   - why it changed
   - how you validated it
6. Request review only after all required checks pass locally.

## Required local checks before PR review

For Python changes, run Ruff, Pyright, and Codacy against the files you changed.

### Ruff

```bash
uv run ruff format src tests
uv run ruff check --fix src tests
uv run pyright
```

### Codacy CLI

Run from repository root and target only `src/` or `tests/`.

```bash
# Analyze specific changed files
.codacy/cli.sh analyze src/path/to/changed_file.py

# Or analyze source tree when needed
.codacy/cli.sh analyze src/
```

If you changed dependencies (`pyproject.toml`), also run:

```bash
.codacy/cli.sh analyze --tool trivy
```

Fix clear code quality and security findings before asking for review.

## Agentic coding requirement

If you are using an AI/agent workflow, you must follow [AGENTS.md](AGENTS.md).

In particular:

- respect skill routing and project conventions
- run Ruff, Pyright, and Codacy on touched files
- resolve Pylint, Semgrep, and Lizard findings
- resolve HIGH/CRITICAL Trivy CVEs before introducing new dependencies

## Style and conventions

- Follow Python and Dagster standards documented in [AGENTS.md](AGENTS.md).
- Use clear names, type annotations, and concise docstrings.
- Avoid unrelated refactors in feature/fix PRs.
