# Agent Guidelines — hsds-entity-resolution

## Code Quality

After every Python file edit, run Ruff first, then Codacy analysis from the project root:

```bash
# Format and auto-fix issues first
uv run ruff format src tests
uv run ruff check --fix src tests

# Analyze a specific file
.codacy/cli.sh analyze src/path/to/file.py

# Analyze all source code
.codacy/cli.sh analyze src/

# Scan dependencies for vulnerabilities after changing pyproject.toml
.codacy/cli.sh analyze --tool trivy
```

Always target `src/` or `tests/` for Ruff/Codacy — never the project root. `.venv/` and other non-source paths are excluded via `.codacy/codacy.yaml` and must never be analyzed.

Fix all Ruff, Pylint, Semgrep, and Lizard findings before finishing. Resolve HIGH/CRITICAL Trivy CVEs before adding any new dependency.

---

## Python Standards

- **Naming:** `PascalCase` classes, `snake_case` functions/variables, `UPPER_SNAKE_CASE` constants
- **Docstrings:** every public module, class, and function
- **Type annotations:** all function parameters and return types
- **Imports:** stdlib → third-party → local, each group separated by a blank line; no wildcard imports
- **Complexity:** max cyclomatic complexity 10 per function; max 50 lines per function
- **Exceptions:** always catch specific exceptions; never use bare `except:`; never silently swallow errors
- **Security:** no `eval()`, `exec()`, or `pickle.loads()` on untrusted data; no hardcoded secrets

## Skills

Reusable agent skills live in `.agents/skills/`. Read a skill's `SKILL.md` before starting any task it covers — skills contain routing tables, quick-reference commands, and links to detailed sub-documents that you would otherwise have to rediscover manually.

### Available skills

| Skill | `SKILL.md` path | When to invoke |
| ----- | --------------- | -------------- |
| **dagster-expert** | `.agents/skills/dagster-expert/SKILL.md` | Any task involving assets, schedules, sensors, jobs, components, `dg` CLI commands, or Dagster project structure. Invoke *before* exploring the codebase. |
| **dignified-python** | `.agents/skills/dignified-python/SKILL.md` | Writing, reviewing, or refactoring Python — type annotations, exception handling, pathlib, ABC/Protocol interfaces, CLI patterns, code quality. |

### How to use a skill

1. **Read the `SKILL.md` first.** It contains a task router and tells you which sub-documents to load for your specific task.
2. **Follow the conditional-loading instructions.** Each skill only requires you to read additional reference files when the task triggers them — avoid loading everything upfront.
3. **Invoke early.** For Dagster work, read `dagster-expert/SKILL.md` before touching any code. For Python standards questions, read `dignified-python/SKILL.md` before writing or reviewing code.
4. **Chain skills when needed.** A task can require both skills — e.g., scaffolding a new Dagster asset (dagster-expert) and then refactoring the generated Python (dignified-python).

---

## Dagster Patterns

- Component classes must inherit from `dg.Component`, `dg.Model`, and `dg.Resolvable`
- `build_defs` must return `dg.Definitions` and take `context: dg.ComponentLoadContext`
- Name assets and ops to reflect the data transformation they perform
