# Repository Guidelines

This repository currently contains a single mailbox export file and no application source code or build tooling. Treat it as a data-only repo unless additional files are added later.

## Project Structure & Module Organization

- `All mail Including Spam and Trash-002.mbox`: Raw mailbox data at the repository root.
- No `src/`, `tests/`, or configuration directories are present at this time.

If you add code, keep it organized (for example, `src/` for parsing logic, `tests/` for automated checks, and `scripts/` for one-off utilities).

## Build, Test, and Development Commands

No build, run, or test commands are defined in the repository. If you introduce code, document the exact commands here (for example, `python -m pytest` or `npm test`) and keep them aligned with any added tooling.

## Coding Style & Naming Conventions

There are no established formatting or linting rules in this repo. If you add scripts or code, specify:

- Indentation (for example, 2 spaces for JavaScript, 4 spaces for Python).
- Naming patterns (for example, `snake_case` for files, `PascalCase` for classes).
- Any formatters/linters you introduce (for example, `black`, `ruff`, `eslint`).

## Testing Guidelines

No testing framework is configured. If tests are added, document the framework, test directory (for example, `tests/`), and naming conventions (for example, `test_*.py` or `*.spec.js`).

## Commit & Pull Request Guidelines

No Git history is present in this repository, so no commit conventions are discoverable. If contributing:

- Use clear, imperative commit messages (for example, “Add mbox parser”).
- Include a brief PR description that summarizes changes and notes any data handling considerations.

## Security & Data Handling

The `.mbox` file likely contains sensitive email data. Avoid committing derived artifacts that expose personal information, and sanitize any sample data used in documentation or tests.
