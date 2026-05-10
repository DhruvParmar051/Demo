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


class AttrDict(dict):
    """
    A helper class that allows dictionary keys to be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No such configuration key: {key}")

    def get_path(self, dotted_path: str, default: Any = None) -> Any:
        """
        Helper to safely get nested keys using a dot string.
        """
        items = dotted_path.split(".")
        val = self

        for item in items:
            if isinstance(val, dict) and item in val:
                val = val[item]
            else:
                return default

        return val


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Loads a JSONL file into a list of dictionaries.
    """
    if not path or not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def train(cfg: Any = None) -> dict[str, Any]:
    """
    Main training loop for the AlphaNetwork.
    """

    # ------------------------------------------------------------------
    # Load and normalize config
    # ------------------------------------------------------------------

    raw_cfg = cfg if cfg is not None else get_config()

    # Convert Pydantic config into a regular dictionary
    # so AttrDict utilities work properly.
    if hasattr(raw_cfg, "model_dump"):
        raw_cfg = raw_cfg.model_dump()

    cfg = AttrDict(raw_cfg)

    set_seed(42)

    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # type: ignore

    except ImportError as exc:
        logger.error("torch required: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    from src.cgal.alpha_network import AlphaNetwork
    from src.retrieval.vector_store import ChromaVectorStore

    # ------------------------------------------------------------------
    # Load alpha supervision labels
    # ------------------------------------------------------------------

    label_path_str = (
        cfg.get_path("data.synthetic.alpha_labels_path")
        or cfg.get_path("data.alpha_labels_path")
        or "data/synthetic/alpha_labels_v2.jsonl"
    )

    labels = _load_jsonl(Path(label_path_str))

    if not labels:
        logger.error(f"Labels empty or not found at: {label_path_str}")

        if hasattr(cfg, "data"):
            logger.info(
                f"Available keys in 'data': {list(cfg.data.keys())}"
            )

        return {"status": "skipped", "reason": "no_labels"}

    logger.info("Loaded %d alpha supervision labels.", len(labels))

    # ------------------------------------------------------------------
    # Initialize encoder + alpha network
    # ------------------------------------------------------------------

    encoder = ChromaVectorStore().model
    net = AlphaNetwork()

    # ------------------------------------------------------------------
    # Dataset wrapper
    # ------------------------------------------------------------------

    class _Dataset(Dataset):
        """
        Simple dataset wrapper around alpha label records.
        """

        def __init__(self, records: list[dict[str, Any]]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, i: int) -> dict[str, Any]:
            return self.records[i]

    # ------------------------------------------------------------------
    # Batch collation
    # ------------------------------------------------------------------

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert raw batch records into model-ready tensors.
        """

        queries = [b["query"] for b in batch]

        alphas = torch.tensor(
            [float(b["optimal_alpha"]) for b in batch],
            dtype=torch.float32,
        )

        # Generate dense embeddings
        embs = encoder.encode(
            queries,
            normalize_embeddings=True,
        )

        # Extract alpha-network feature vectors
        feats = [
            net.extract_features(q, e, "")
            for q, e in zip(queries, embs)
        ]

        feats = torch.stack(feats, dim=0)

        return {
            "features": feats,
            "alpha": alphas,
        }

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------

    bs = int(cfg.get_path("training.alpha.batch_size", 32))
    lr = float(cfg.get_path("training.alpha.learning_rate", 3e-4))
    epochs = int(cfg.get_path("training.alpha.epochs", 30))

    logger.info(
        "Alpha training config | batch_size=%d | lr=%f | epochs=%d",
        bs,
        lr,
        epochs,
    )

    # ------------------------------------------------------------------
    # Train / validation split
    # ------------------------------------------------------------------

    import random
    random.shuffle(labels)
    split = int(0.9 * len(labels))
    train_records = labels[:split]
    val_records = labels[split:]

    # ------------------------------------------------------------------
    # Weighted sampler — oversample non-0.5 labels to balance classes
    # ------------------------------------------------------------------

    alpha_vals = [float(r["optimal_alpha"]) for r in train_records]
    # Bucket alphas into 5 bins: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    def _bucket(a: float) -> int:
        return min(int(a * 5), 4)

    from collections import Counter as _Counter
    bucket_counts = _Counter(_bucket(a) for a in alpha_vals)
    max_count = max(bucket_counts.values())
    sample_weights = [max_count / bucket_counts[_bucket(a)] for a in alpha_vals]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_records),
        replacement=True,
    )

    train_loader = DataLoader(
        _Dataset(train_records),
        batch_size=bs,
        sampler=sampler,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        _Dataset(val_records),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate,
    )

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------

    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs, eta_min=lr * 0.01
    )

    # ------------------------------------------------------------------
    # Training loop  (Huber loss — robust to tie-label outliers)
    # ------------------------------------------------------------------

    logger.info("Starting AlphaNetwork training for %d epochs...", epochs)

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):

        net.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            pred = net(batch["features"]).squeeze(-1)
            loss = F.huber_loss(pred, batch["alpha"], delta=0.15)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()
            steps += 1

        avg_train = total_loss / max(steps, 1)
        scheduler.step()

        # Validation
        net.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = net(batch["features"]).squeeze(-1)
                val_loss += F.huber_loss(pred, batch["alpha"], delta=0.15).item()
                val_steps += 1
        avg_val = val_loss / max(val_steps, 1)

        if avg_val < best_loss:
            best_loss = avg_val
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        logger.info(
            "Epoch %d/%d | train=%.4f | val=%.4f | lr=%.2e",
            epoch + 1, epochs, avg_train, avg_val,
            scheduler.get_last_lr()[0],
        )

    # Restore best weights
    if best_state is not None:
        net.load_state_dict(best_state)
        logger.info("Restored best weights (val_loss=%.4f)", best_loss)

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------

    checkpoint_dir_path = cfg.get_path(
        "checkpoints.alpha_network",
        "checkpoints/alpha_network",
    )

    out_dir = Path(checkpoint_dir_path)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = out_dir / "model.pt"

    # Use AlphaNetwork.save() which persists {"state_dict": ..., "config": ...}
    # so AlphaNetwork.load() can reconstruct the model correctly at inference.
    net.save(save_path)

    logger.info(
        "Saved alpha network checkpoint to %s",
        save_path,
    )

    return {
        "status": "ok",
        "best_loss": best_loss,
        "output_dir": str(save_path),
    }


def main() -> None:
    """
    Entrypoint for standalone execution.
    """

    logging.basicConfig(level=logging.INFO)

    result = train()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()