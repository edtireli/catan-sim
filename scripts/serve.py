#!/usr/bin/env python3
"""Launch the Catan web server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8080
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start Catan web server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"Starting Catan server at http://localhost:{args.port}")
    print("Open in your browser to play!")
    print()

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
