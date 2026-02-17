#!/usr/bin/env python3
"""
Copy files listed in a CSV column from source_dir to dest_dir.

- Strips paths from CSV values (uses basename only)
- Copies only those files listed
- Sets destination file mode to 775
- Supports multiprocessing (can be slower on sshfs; disable with --no-mp)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple, Optional

import pandas as pd


MODE_775 = 0o775


def _copy_one(args: Tuple[str, str, str, bool]) -> Tuple[str, bool, str]:
    """
    Worker-safe function.
    Returns: (filename, success, message)
    """
    filename, source_dir, dest_dir, overwrite = args
    try:
        src = Path(source_dir) / filename
        dst = Path(dest_dir) / filename

        if not src.exists():
            return filename, False, f"missing in source: {src}"

        if dst.exists() and not overwrite:
            return filename, True, f"exists, skipped: {dst}"

        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # copy2 preserves metadata (times). On network shares this is fine but not required.
        shutil.copy2(src, dst)

        # Set permissions to 775 (note: sshfs mount options/umask may override)
        os.chmod(dst, MODE_775)

        return filename, True, "copied"
    except Exception as e:
        return filename, False, f"error: {type(e).__name__}: {e}"


def _load_filenames(csv_path: str, column: str) -> list[str]:
    df = pd.read_csv(csv_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Columns: {list(df.columns)}")

    # Drop NaN, cast to str, strip whitespace, take basename, drop empties, dedupe (preserve order)
    series = (
        df[column]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .map(lambda s: os.path.basename(s))
    )

    seen = set()
    out: list[str] = []
    for name in series:
        if not name or name in (".", ".."):
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _pick_executor(kind: str, max_workers: int):
    """
    On sshfs, threads often perform as well or better than processes for I/O-bound copying,
    and are cheaper. Processes can still help if python overhead is significant, but usually isn’t.
    """
    if kind == "process":
        return ProcessPoolExecutor(max_workers=max_workers)
    if kind == "thread":
        return ThreadPoolExecutor(max_workers=max_workers)
    raise ValueError("executor kind must be 'thread' or 'process'")


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV")
    p.add_argument("--column", required=True, help="CSV column containing file paths/names")
    p.add_argument("--source-dir", required=True, help="Source directory")
    p.add_argument("--dest-dir", required=True, help="Destination directory")
    p.add_argument("--overwrite", action="store_true", help="Overwrite if destination exists")
    p.add_argument("--max-workers", type=int, default=8, help="Worker count (default 8)")
    p.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help="Concurrency type. Default thread (usually best for sshfs).",
    )
    p.add_argument(
        "--no-mp",
        action="store_true",
        help="Disable concurrency, copy sequentially",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be copied",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    csv_path = str(Path(args.csv))
    source_dir = str(Path(args.source_dir))
    dest_dir = str(Path(args.dest_dir))

    filenames = _load_filenames(csv_path, args.column)
    if not filenames:
        print("No filenames found after cleaning input column.", file=sys.stderr)
        return 2

    # Optional: quick source existence check can be expensive on sshfs; do it per-file in worker.
    tasks = [(fn, source_dir, dest_dir, args.overwrite) for fn in filenames]

    if args.dry_run:
        for fn in filenames:
            print(f"would copy: {Path(source_dir)/fn} -> {Path(dest_dir)/fn}")
        return 0

    ok = 0
    skipped_or_ok = 0
    failed = 0

    if args.no_mp or args.max_workers <= 1:
        for t in tasks:
            fn, success, msg = _copy_one(t)
            if success:
                skipped_or_ok += 1
                if msg == "copied":
                    ok += 1
            else:
                failed += 1
            print(f"{fn}\t{msg}")
    else:
        with _pick_executor(args.executor, args.max_workers) as ex:
            futures = [ex.submit(_copy_one, t) for t in tasks]
            for fut in as_completed(futures):
                fn, success, msg = fut.result()
                if success:
                    skipped_or_ok += 1
                    if msg == "copied":
                        ok += 1
                else:
                    failed += 1
                print(f"{fn}\t{msg}")

    print(
        f"\nSummary: listed={len(filenames)} copied={ok} ok_or_skipped={skipped_or_ok} failed={failed}",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
