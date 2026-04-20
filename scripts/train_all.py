"""Orchestrator: trains all 7 components in dependency order.

Skips any component whose checkpoint directory already contains files,
unless ``--force`` is passed.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --force --components generator,dpo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import TRAINERS  # noqa: E402
from src.utils.config import get_config  # noqa: E402

_ORDER = ["retriever", "reranker", "generator", "dpo",
          "confidence", "alpha", "decomposer"]


def _ckpt_dir_for(cfg, component: str) -> Path:
    mapping = {
        "retriever": cfg.checkpoints.retriever,
        "reranker": cfg.checkpoints.reranker,
        "generator": cfg.checkpoints.generator_sft,
        "dpo": cfg.checkpoints.generator_dpo,
        "confidence": cfg.checkpoints.confidence_head,
        "alpha": cfg.checkpoints.alpha_network,
        "decomposer": cfg.checkpoints.decomposer,
    }
    return Path(mapping[component])


def _has_artifacts(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", default=None,
                         help="Comma-separated subset (default: all).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = get_config()

    wanted = _ORDER if args.components is None else [
        c.strip() for c in args.components.split(",") if c.strip()
    ]
    unknown = [c for c in wanted if c not in TRAINERS]
    if unknown:
        raise SystemExit(f"Unknown components: {unknown}")

    results: dict[str, dict] = {}
    for comp in _ORDER:
        if comp not in wanted:
            continue
        out = _ckpt_dir_for(cfg, comp)
        if not args.force and _has_artifacts(out):
            logging.info("Skipping %s -- artifacts already at %s", comp, out)
            results[comp] = {"status": "skipped", "reason": "checkpoint_exists"}
            continue
        logging.info("=== Training %s ===", comp)
        try:
            results[comp] = TRAINERS[comp](cfg)
        except Exception as exc:
            logging.exception("Training %s failed", comp)
            results[comp] = {"status": "error", "error": str(exc)}

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
