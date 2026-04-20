"""Deterministic train/val/test splitter for synthetic JSONL datasets.

This is a TRAINING-TIME utility only. At deployment time, user-uploaded
documents are chunked and indexed directly -- no splitting happens.

Usage:
    python scripts/split_data.py
    python scripts/split_data.py --input-dir data/synthetic --ratios 0.8 0.1 0.1
    python scripts/split_data.py --seed 42 --files qa_pairs.jsonl preferences.jsonl

Behavior:
    - Reads every *.jsonl file directly inside --input-dir (non-recursive).
    - Shuffles each file deterministically with --seed.
    - Splits into train / val / test by --ratios (must sum to 1.0).
    - Writes outputs to <input-dir>/train/<name>.jsonl, .../val/, .../test/.
    - Idempotent: re-running with the same seed and inputs produces identical
      outputs.
    - Emits a manifest at <input-dir>/split_manifest.json with per-file counts
      and the seed / ratios used, so the split is auditable.

Typical pipeline:
    python run.py generate-data --type all
    python scripts/split_data.py
    python run.py train --component all
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_RATIOS = (0.8, 0.1, 0.1)
_SPLIT_NAMES = ("train", "val", "test")


def _read_jsonl(path: Path) -> List[str]:
    """Return the raw non-empty lines of a JSONL file (preserves trailing newline-free lines)."""
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.rstrip("\n")
            if stripped.strip():
                lines.append(stripped)
    return lines


def _write_jsonl(path: Path, lines: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)
            fh.write("\n")
            count += 1
    return count


def _validate_ratios(ratios: Sequence[float]) -> None:
    if len(ratios) != 3:
        raise ValueError("--ratios must have exactly 3 values (train val test)")
    if any(r < 0 for r in ratios):
        raise ValueError("--ratios values must be non-negative")
    total = sum(ratios)
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"--ratios must sum to 1.0 (got {total:.4f})")


def _compute_boundaries(n: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    """Compute (n_train, n_val, n_test) so that all items are placed.

    Rounds train and val down, then assigns the remainder to test. For very
    small n, guarantees val/test get at least one item when their ratio > 0
    and there is room.
    """
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    # Guard against the val/test getting zero when they should have one.
    if n >= 3:
        if ratios[1] > 0 and n_val == 0:
            n_val = 1
            n_train = max(0, n_train - 1)
        if ratios[2] > 0 and n_test == 0:
            n_test = 1
            n_train = max(0, n_train - 1)
    return n_train, n_val, n_test


def split_file(
    src: Path,
    input_dir: Path,
    seed: int,
    ratios: Sequence[float],
) -> dict:
    """Split a single JSONL file into train/val/test under <input_dir>/<split>/."""
    lines = _read_jsonl(src)
    n = len(lines)
    if n == 0:
        logger.warning("Skipping empty file: %s", src.name)
        return {"file": src.name, "status": "empty", "counts": {"train": 0, "val": 0, "test": 0}}

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    shuffled = [lines[i] for i in indices]

    n_train, n_val, n_test = _compute_boundaries(n, ratios)
    train_lines = shuffled[:n_train]
    val_lines = shuffled[n_train : n_train + n_val]
    test_lines = shuffled[n_train + n_val :]

    counts = {}
    for split_name, split_lines in zip(_SPLIT_NAMES, (train_lines, val_lines, test_lines)):
        out_path = input_dir / split_name / src.name
        counts[split_name] = _write_jsonl(out_path, split_lines)

    logger.info(
        "Split %s: total=%d -> train=%d val=%d test=%d",
        src.name, n, counts["train"], counts["val"], counts["test"],
    )
    return {"file": src.name, "status": "ok", "total": n, "counts": counts}


def split_directory(
    input_dir: Path,
    seed: int = 42,
    ratios: Sequence[float] = _DEFAULT_RATIOS,
    file_whitelist: Sequence[str] | None = None,
) -> dict:
    """Split every top-level JSONL in input_dir into train/val/test subdirs.

    Returns a manifest dict. Writes <input_dir>/split_manifest.json.
    """
    _validate_ratios(ratios)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    jsonl_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix == ".jsonl")
    if file_whitelist:
        wanted = set(file_whitelist)
        jsonl_files = [p for p in jsonl_files if p.name in wanted]
    if not jsonl_files:
        logger.warning("No JSONL files found in %s", input_dir)

    entries = [split_file(p, input_dir, seed, ratios) for p in jsonl_files]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "files": entries,
    }
    manifest_path = input_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote manifest: %s", manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="data/synthetic",
                        help="Directory holding generated *.jsonl files.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic shuffling.")
    parser.add_argument("--ratios", nargs=3, type=float, default=list(_DEFAULT_RATIOS),
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Three floats summing to 1.0.")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Optional whitelist of filenames (basenames) to split.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        manifest = split_directory(
            Path(args.input_dir),
            seed=args.seed,
            ratios=tuple(args.ratios),
            file_whitelist=args.files,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
