# Repository Guidelines

## Project Structure & Module Organization

`pic_utils/` contains the Python package. Top-level modules group utilities by domain, such as `hdf5.py`, `spectral.py`, `plasma.py`, `geometry.py`, and `materials.py`. FBPIC-specific helpers live under `pic_utils/fbpic/`. Tests are in `tests/` and generally mirror package modules. Shared test helpers belong in `tests/utils.py`.

Metadata and tooling are configured in `pyproject.toml`, `ruff.toml`, and `pyrightconfig.json`. Runtime dependencies are listed in `requirements.txt`.

## Build, Test, and Development Commands

Use the provided conda environment for local development.

```bash
conda activate <provided-env>
```

Use these commands during development:

```bash
pytest
ruff check .
ruff format .
pyright
```

`pytest` runs the full test suite with the repository root on `PYTHONPATH`. `ruff check .` reports style and lint issues, while `ruff format .` applies the configured formatter. `pyright` is configured but type checking is currently disabled, so treat it as a lightweight sanity check.

## Coding Style & Naming Conventions

Write Python using Ruff formatting with a 120-character line length and single quotes. Follow existing module style: small domain-focused functions, clear numerical variable names, and explicit unit handling. Use `snake_case` for functions, variables, modules, and test files; use `PascalCase` only for classes.

Keep public APIs importable without heavy side effects. Prefer NumPy/SciPy vectorized operations over ad hoc loops.

## Testing Guidelines

Update tests in `tests/` for behavioral changes. Name files `test_<module>.py` and test functions `test_<behavior>()`. Use deterministic numeric assertions and scientific tolerances, for example `numpy.testing.assert_allclose`. Don't create tests for modules which don't have tests already, unless explicitly asked.

Run `pytest` before submitting changes. For focused work, run a single module such as:

```bash
pytest tests/test_spectral.py
```

## Commit & Pull Request Guidelines

Commits use short imperative messages, for example `Implement xy fields reading`. Keep commits focused and describe the user-visible or API-level change.

Pull requests should include a concise description, the reason for the change, and the tests or checks run. Link related issues when available. Include screenshots only for changes that affect generated plots, diagnostics output, or visual artifacts.

## Agent-Specific Instructions

Keep edits scoped to the requested behavior and avoid unrelated formatting churn. Do not change dependency versions or public APIs unless the task requires it. When modifying numerical routines, add regression tests that cover units, shapes, and representative edge cases.

Machine-specific instruction can be found in ENV.md if it exists.
