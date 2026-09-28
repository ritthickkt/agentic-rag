"""Start the API and UI.

    python scripts/serve.py [--port 8000] [--reload]

A thin wrapper around uvicorn that exists to enforce one thing: **a single worker.**
Embedded Qdrant locks its storage directory to one process, so a second worker would
crash on startup rather than load-balance. `--workers` is therefore not offered; if
you need concurrency, run a Qdrant server and set `QDRANT_URL`, which makes the store
shared and the process count a free choice (DECISIONS D21).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="restart on code changes")
    args = ap.parse_args()

    from agentic_rag.config import CONFIG

    try:
        CONFIG.require_api_key()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    import uvicorn

    print(f"  UI       http://{args.host}:{args.port}")
    print(f"  API docs http://{args.host}:{args.port}/docs")
    print(f"  store    {CONFIG.qdrant_path}")
    print(f"  models   answer={CONFIG.answer_model_id}  embed={CONFIG.embed_model}\n")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # Deliberately absent: workers. See the module docstring.
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
