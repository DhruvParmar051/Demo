"""Evaluate all 8 pipelines (b1-b3, m1-m5) on the test set and generate the report.

Usage:
    python scripts/evaluate_all.py --test-dir data/test --output-dir report
"""

from __future__ import annotations

import argparse
import gc
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


def _free_pipeline(pipeline) -> None:
    """Release model weights and reclaim memory before loading the next model."""
    try:
        import torch
        # Walk common attribute names that hold large tensors.
        for attr in ("generator", "vector_store", "reranker", "confidence_head",
                     "alpha_network", "_model", "_llama", "model", "engine"):
            obj = getattr(pipeline, attr, None)
            if obj is None:
                continue
            # Recurse one level for nested components (e.g. pipeline.generator._model).
            for inner in ("_model", "_llama", "model"):
                inner_obj = getattr(obj, inner, None)
                if inner_obj is not None:
                    try:
                        inner_obj.cpu()
                    except Exception:
                        pass
                    setattr(obj, inner, None)
            try:
                obj.cpu()
            except Exception:
                pass
            setattr(pipeline, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default="data/test/qa_pairs.jsonl")
    parser.add_argument("--output-dir", default="report")
    parser.add_argument("--models", default=",".join(_TAGS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = get_config()

    tags = [t.strip().lower() for t in args.models.split(",") if t.strip()]
    evaluator = Evaluator(Path(args.test_dir), Path(args.output_dir))

    # Evaluate one model at a time to avoid holding all weights in memory.
    aggregate_results: dict[str, dict] = {}
    _seen_gguf_paths: dict[str, str] = {}  # gguf_path -> first tag that used it
    for tag in tags:
        logging.info("=== Loading model: %s ===", tag)
        try:
            pipeline = _build_pipeline(tag, cfg)
        except Exception as exc:
            logging.warning("Failed to build %s: %s", tag, exc)
            continue

        try:
            res = evaluator.evaluate(tag, pipeline)
            aggregate_results[tag] = res["aggregate"]
            # Record which checkpoint was actually loaded so collapsed-GGUF
            # runs are detectable in the output JSON.
            gguf_path = getattr(getattr(pipeline, "generator", None), "gguf_path", None)
            if gguf_path:
                gguf_path = str(gguf_path)
                res["gguf_path"] = gguf_path
                if gguf_path in _seen_gguf_paths:
                    logging.warning(
                        "COLLAPSED GGUF: %s and %s are loading the same checkpoint (%s). "
                        "Their results will be identical. "
                        "Run scripts/convert_to_gguf.py --variant <base|sft|dpo> to produce "
                        "per-variant GGUF files.",
                        _seen_gguf_paths[gguf_path], tag, gguf_path,
                    )
                else:
                    _seen_gguf_paths[gguf_path] = tag
            if hasattr(pipeline, "flags"):
                res["flags"] = {k: v for k, v in vars(pipeline.flags).items()}
            # Save per-model result immediately so progress is not lost on crash.
            per_model_path = Path(args.output_dir) / f"{tag}.json"
            per_model_path.parent.mkdir(parents=True, exist_ok=True)
            with per_model_path.open("w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, default=str)
            logging.info("Saved %s results to %s", tag, per_model_path)
        except Exception as exc:
            logging.warning("Evaluation failed for %s: %s", tag, exc)
        finally:
            logging.info("Freeing memory for %s ...", tag)
            _free_pipeline(pipeline)
            del pipeline

    out_path = Path(args.output_dir) / "all_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate_results, f, indent=2, default=str)
    logging.info("Wrote %s", out_path)

    generate_report(out_path, Path(args.output_dir))
    logging.info("Report generated in %s", args.output_dir)


if __name__ == "__main__":
    main()
