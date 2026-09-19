# Contributing

## Development setup

```bash
git clone https://github.com/ustunb/riskslim
cd riskslim
uv sync --group dev
```

CPLEX is a required runtime dependency for solver work and the test suite.

## Tests

```bash
uv run pytest
uv run pytest -m slow
uv run pytest -m ""
# Run the standalone CPLEX correctness check.
uv run pytest --noconftest tests/test_cplex.py -q
```

New behavior needs a test. Bug fixes need a test that fails before the fix.

## Before opening a pull request

```bash
uv run ruff check riskslim tests
uv run ruff format --check examples/quickstart.py tests/conftest.py \
  tests/generate_training_test_cases.py tests/test_training.py \
  tests/test_training_known.py tests/test_cplex.py
uv build && uvx twine check dist/*
```

CI runs the same checks, plus an install check that runs the test suite and
`examples/quickstart.py` against the built wheel and sdist rather than the source tree.

## Conventions

- Work happens on `dev`; `main` is what is released.
- Google-style docstrings on public API functions and classes.
- Runtime dependencies are declared in `pyproject.toml` — CI installs from the built
  artifact, so an undeclared import fails there even when it works locally.
- CPLEX is required and is declared as a regular dependency.
