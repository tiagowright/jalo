# Repository Guidelines

## Project Structure & Module Organization
- Core logic lives in `keyboard_model.py`; extend this module to expose new metrics or layout utilities.
- Hardware definitions sit in `hardware.py` and `keebs/`; each board module must export a `KEYBOARD` instance.
- Frequency data and corpora are under `corpus/` and `freqdist.py`; keep large datasets in `assets/` if added later.
- Add fixtures or regression data to `tests/data/`, and mirror layout examples in `layouts/` when documenting.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create and enter the project virtual environment.
- `python -m pip install numpy pytest` — install runtime and baseline testing dependencies; pin in `requirements.txt` once stable.
- `python -m pytest` — execute the automated suite; use `-k` for targeted cases and `--cov=keyboard_model --cov-report=term-missing` to inspect coverage gaps.
- `python -m pip install -e .` — optional editable install after adding packaging metadata.

## Coding Style & Naming Conventions
- Target Python 3.11+ with four-space indentation and type hints on public call signatures (e.g., `def score(self, layout: np.ndarray) -> float:`).
- Follow NumPy-style docstrings for exported functions and classes; keep module-level comments concise.
- Use domain-aligned names (`F1`, `T2`, `swap_delta`) and snake_case helpers (`transform_layout`).
- Prefer `black` and `ruff` for formatting/linting when available; document command invocations in this file when tooling changes.

## Testing Guidelines
- Tests live in `tests/`; begin with `tests/test_keyboard_model.py` and expand coverage alongside new features.
- Name tests `test_<behavior>` and include regression IDs in docstrings when referencing bugs.
- Aim for full branch coverage on swap logic and new metrics; run `pytest --cov=keyboard_model --cov-report=term-missing` before opening a PR.
- Store fixtures in `tests/data/` with inline comments describing schema and provenance.

## Commit & Pull Request Guidelines
- Write small, focused commits in imperative mood (e.g., `Add delta swap benchmark`), referencing issues in the footer (`Refs #12`) when relevant.
- Ensure CI (tests + linters) is green before requesting review; mention any deliberate skips or TODOs in the PR description.
- Summarize the motivating problem, list key changes, and attach metrics or screenshots for performance-affecting work.
- Call out data dependencies, follow-up tasks, and testing evidence so reviewers can verify quickly.
