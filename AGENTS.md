# Repository Guidelines

## Project Structure & Module Organization
`keyboard_model.py` holds the entire public API. It defines the `Metric` dataclass plus the `KeyboardModel` class that wraps metric aggregation, layout scoring, and swap deltas. Keep auxiliary helpers adjacent to the logic they support and prefer extending this module over creating free-floating scripts. If you need fixtures (datasets, layouts, benchmark outputs), place them under a new `assets/` or `tests/data/` directory and document the format in-file.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment; required before installing dependencies.
- `python -m pip install numpy pytest`: install runtime and testing requirements; pin exact versions in `requirements.txt` when finalized.
- `python -m pytest`: run all automated tests (add the `-k` or `-m` flags to focus on targeted suites).
- `python -m pip install -e .`: once a `pyproject.toml` or `setup.cfg` is added, use editable installs for downstream experiments.

## Coding Style & Naming Conventions
Use Python 3.11+ features, four-space indentation, and type hints on all public call signatures (`np.ndarray`, `Optional[int]`, etc.). Follow NumPy-style docstrings for new public functions and keep module-level documentation concise. Arrays and tensors should be named by domain (`F1`, `T2`, `V3_tot`) to mirror the existing notation; new helpers should use snake_case verbs (`transform_layout`, `validate_metrics`). Run `ruff` or `black` if adopted—record command invocations in this guide for consistency.

## Testing Guidelines
Adopt `pytest` with `tests/test_keyboard_model.py` as the starter suite. Mirror scenarios from the docstring: layout permutations, scoring correctness, and `delta_swap` invariants. Name tests `test_<behavior>` and include regression IDs in docstrings when referencing bugs. Target full branch coverage on swap logic and any new metrics, using `pytest --cov=keyboard_model --cov-report=term-missing` before submitting.

## Commit & Pull Request Guidelines
Commits should be small, focused, and use imperative mood (`Add delta swap benchmark`) as in the existing history. Reference issue numbers in the footer when applicable (`Refs #12`). Pull requests must summarize the motivating problem, list key changes, and call out performance measurements or data dependencies. Attach screenshots or tables when presenting benchmark deltas. Ensure CI (tests + linters) is green before requesting review and note any follow-up tasks explicitly.
