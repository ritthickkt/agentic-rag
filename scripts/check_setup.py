"""Smoke test: does the environment actually work? Run this after install.

Checks each dependency separately so a failure tells you which one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_rag.config import CONFIG


def main() -> int:
    ok = True

    print("--- config ---")
    for k, v in CONFIG.to_dict().items():
        print(f"  {k:20s} {v}")

    print("\n--- imports ---")
    for mod in ("anthropic", "qdrant_client", "sentence_transformers", "datasets", "numpy"):
        try:
            __import__(mod)
            print(f"  ok      {mod}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  FAILED  {mod}: {e}")

    print("\n--- api key ---")
    try:
        key = CONFIG.require_api_key()
        print(f"  ok      ANTHROPIC_API_KEY present (…{key[-4:]})")
    except RuntimeError as e:
        ok = False
        print(f"  FAILED  {e}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
