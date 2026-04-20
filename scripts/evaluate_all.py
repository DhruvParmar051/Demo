"""Evaluate all 8 pipelines (b1-b3, m1-m5) on the test set and generate the report.

Usage:
    python scripts/evaluate_all.py --test-dir data/test --output-dir report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import Evaluator  # noqa: E402
from src.evaluation.report import generate_report  # noqa: E402
from src.models.baselines import BaselineB1, BaselineB2, BaselineB3  # noqa: E402
from src.models.m5_pipeline import M5Pipeline  # noqa: E402
from src.utils.config import get_config  # noqa: E402


_TAGS = ["b1", "b2", "b3", "m1", "m2", "m3", "m4", "m5"]


def _build_pipeline(tag: str, cfg):
    if tag == "b1":
        return BaselineB1()
    if tag == "b2":
        return BaselineB2()
    if tag == "b3":
        return BaselineB3()
    return M5Pipeline.from_tag(tag, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--output-dir", default="report")
    parser.add_argument("--models", default=",".join(_TAGS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = get_config()

    evaluator = Evaluator(Path(args.test_dir), Path(args.output_dir))
    pipelines = {}
    for tag in args.models.split(","):
        tag = tag.strip().lower()
        if not tag:
            continue
        try:
            pipelines[tag] = _build_pipeline(tag, cfg)
        except Exception as exc:
            logging.warning("Failed to build %s: %s", tag, exc)

    results = evaluator.evaluate_all(pipelines)
    out_path = Path(args.output_dir) / "all_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logging.info("Wrote %s", out_path)

    generate_report(out_path, Path(args.output_dir))
    logging.info("Report generated in %s", args.output_dir)


if __name__ == "__main__":
    main()
