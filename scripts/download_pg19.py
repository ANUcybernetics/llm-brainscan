#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=2.20,<3.0"]
# ///
"""Stream PG-19 from HuggingFace and concatenate text into a single binary file.

Run on the deployment machine where storage is plentiful, e.g.:
    mise exec -- uv run scripts/download_pg19.py /ssd/brainscan/data/pg19.bin
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from datasets import load_dataset

BOOK_SEPARATOR = b"\n\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination binary file")
    parser.add_argument(
        "--split", default="train", choices=("train", "validation", "test")
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N books (smoke-test convenience)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "deepmind/pg19",
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )

    t0 = time.time()
    n_books = 0
    n_bytes = 0
    with open(args.output, "wb") as f:
        for record in ds:
            if args.limit is not None and n_books >= args.limit:
                break
            payload = record["text"].encode("utf-8") + BOOK_SEPARATOR
            f.write(payload)
            n_books += 1
            n_bytes += len(payload)
            if n_books % 100 == 0:
                elapsed = time.time() - t0
                rate = n_bytes / elapsed / 1e6
                print(
                    f"  {n_books:,} books | {n_bytes / 1e9:.2f} GB"
                    f" | {rate:.1f} MB/s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"Done: {n_books:,} books, {n_bytes / 1e9:.2f} GB"
        f" in {elapsed:.0f}s -> {args.output}"
    )


if __name__ == "__main__":
    main()
