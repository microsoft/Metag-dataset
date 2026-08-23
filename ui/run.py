#!/usr/bin/env python3
"""Launch the browser-based diff annotation UI.

    python run.py                                    # runs the bundled example batch
    python run.py my_batch.jsonl /path/to/pdfs -o out.jsonl
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from webapp.server import build_session, create_app

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "examples" / "anisundar_batch_first10.jsonl"
DEFAULT_PDF_DIR = BASE_DIR / "examples" / "pdfs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_jsonl", nargs="?", default=str(DEFAULT_INPUT), help="JSONL file of action items")
    parser.add_argument("pdf_dir", nargs="?", default=str(DEFAULT_PDF_DIR), help="Directory holding the paper PDFs")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL (default: <input>_with_diffs.jsonl)")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (use 0.0.0.0 to share on a LAN)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window automatically")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    if not input_jsonl.is_file():
        print(f"Error: input file not found: {input_jsonl}", file=sys.stderr)
        return 1
    if not pdf_dir.is_dir():
        print(f"Error: PDF directory not found: {pdf_dir}", file=sys.stderr)
        return 1

    output_path = Path(args.output).resolve() if args.output else input_jsonl.with_name(f"{input_jsonl.stem}_with_diffs.jsonl")
    session = build_session(input_jsonl, pdf_dir, output_path, BASE_DIR / ".cache" / "diffs")

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    print(f"Input:  {input_jsonl}")
    print(f"PDFs:   {pdf_dir}")
    print(f"Output: {output_path}")
    print(f"Entries: {session.index} already annotated, {len(session.entries) - session.index} remaining")
    print(f"\nOpen {url} in your browser (Ctrl+C to stop).\n")

    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    create_app(session).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
