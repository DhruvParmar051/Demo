"""
Train the AlphaNetwork MLP that predicts the optimal dense/sparse fusion
weight per query. Supervised by oracle alpha labels produced by
``src/data/alpha_label_generator.py`` via an alpha-grid sweep.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def train(cfg: Any = None) -> dict[str, Any]:
    cfg = cfg if cfg is not None else get_config()
    set_seed(42)

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
    except ImportError as exc:
        logger.error("torch required: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    from src.cgal.alpha_network import AlphaNetwork
    from src.retrieval.vector_store import ChromaVectorStore

    labels = _load_jsonl(Path(cfg.data.synthetic.alpha_labels_path))
    if not labels:
        return {"status": "skipped", "reason": "no_labels"}

    encoder = ChromaVectorStore().model
    net = AlphaNetwork()

    class _Dataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, i: int) -> dict[str, Any]:
            return self.records[i]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        queries = [b["query"] for b in batch]
        alphas = torch.tensor(
            [float(b["optimal_alpha"]) for b in batch], dtype=torch.float32
        )
        embs = encoder.encode(queries, normalize_embeddings=True)
        feats = [net.extract_features(q, e, "") for q, e in zip(queries, embs)]
        feats = torch.stack(feats, dim=0)
        return {"features": feats, "alpha": alphas}

    bs = int(cfg.training.alpha.batch_size)
    lr = float(cfg.training.alpha.learning_rate)
    epochs = int(cfg.training.alpha.epochs)

    split = int(0.9 * len(labels))
    tr, va = labels[:split], labels[split:]
    tr_loader = DataLoader(_Dataset(tr), batch_size=bs, shuffle=True,
                           collate_fn=collate)

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        tot, n = 0.0, 0
        for batch in tr_loader:
            pred = net(batch["features"]).squeeze(-1)
            loss = F.mse_loss(pred, batch["alpha"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item()
            n += 1
        logger.info("Epoch %d: loss=%.4f", epoch + 1, tot / max(n, 1))

    out_dir = Path(cfg.checkpoints.alpha_network)
    out_dir.mkdir(parents=True, exist_ok=True)
    net.save(out_dir)
    logger.info("Saved alpha network to %s", out_dir)
    return {"status": "ok", "output_dir": str(out_dir)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()
