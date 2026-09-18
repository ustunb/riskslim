"""Maintain the checked-in training test-case store.

Known-reference expectations are never recomputed with the solver under test.
Future synthetic generation can load this store and replace only
``synthetic_oracles``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

SCHEMA_VERSION = 1
TESTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = TESTS_DIR / "training_test_cases.pkl"


def load_store(output_path: Path) -> dict:
    """Load the store so later generation can preserve known records."""
    with output_path.open("rb") as file_handle:
        store = pickle.load(file_handle)
    if not isinstance(store, dict) or store.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported training test-case store: {output_path}")
    if not isinstance(store.get("synthetic_oracles"), dict):
        raise ValueError("synthetic_oracles must be a dictionary")
    if not isinstance(store.get("known_reference_results"), dict):
        raise ValueError("known_reference_results must be a dictionary")
    return store


def parse_args() -> argparse.Namespace:
    """Parse the test-case store path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Validate the store without changing its known-reference records."""
    args = parse_args()
    output_path = args.output.resolve()
    if not output_path.exists():
        raise FileNotFoundError(
            "known expectations cannot be recomputed; restore the checked-in store with "
            "`git restore tests/training_test_cases.pkl`"
        )
    store = load_store(output_path)
    print(
        f"validated {output_path}; preserved "
        f"{len(store['known_reference_results'])} known-reference records"
    )


if __name__ == "__main__":
    main()
