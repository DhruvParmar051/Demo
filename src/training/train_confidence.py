"""
Train the joint soft-label confidence head + tool policy head.

Soft labels are BERTScore-derived continuous targets in [0, 1], produced by
``src/data/confidence_label_generator.py``. Tool-routing labels come from
``src/data/decomp_label_generator.py`` (or ToolRouteLabel records).

Loss = MSE(sigmoid(logit) - soft_label) + 0.5 * CE(tool_logits, gold_tool).

After training, applies temperature scaling on a held-out validation split
and saves the resulting ``temperature`` scalar alongside the state_dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


_TOOL_NAMES = ("AnswerDirect", "SearchKB", "GetPolicy", "CreateTicket")


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
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
    except ImportError as exc:
        logger.error("torch required: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    from src.cgal.confidence_head import ConfidenceHead
    from src.evaluation.calibration import compute_ece
    from src.retrieval.vector_store import ChromaVectorStore

    labels = _load_jsonl(Path(cfg.data.synthetic.confidence_labels_path))
    tool_labels = _load_jsonl(
        Path(cfg.data.synthetic.tool_route_labels_path)
    ) if hasattr(cfg.data.synthetic, "tool_route_labels_path") else []
    if not labels:
        return {"status": "skipped", "reason": "no_labels"}

    tool_lookup = {t.get("query"): t.get("gold_tool", "AnswerDirect")
                   for t in tool_labels}

    vstore = ChromaVectorStore()
    encoder = vstore.model

    class _Dataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, i: int) -> dict[str, Any]:
            return self.records[i]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        queries = [b["query"] for b in batch]
        soft = torch.tensor([float(b["soft_label"]) for b in batch], dtype=torch.float32)
        # Gold tool indices (defaults to AnswerDirect).
        tools = torch.tensor(
            [
                _TOOL_NAMES.index(tool_lookup.get(b["query"], "AnswerDirect"))
                if tool_lookup.get(b["query"], "AnswerDirect") in _TOOL_NAMES
                else 0
                for b in batch
            ],
            dtype=torch.long,
        )
        q_emb = torch.tensor(
            encoder.encode(queries, normalize_embeddings=True),
            dtype=torch.float32,
        )
        # Evidence embeddings: simple zero-fill when we don't have cached text.
        dim = q_emb.size(-1)
        e_emb = torch.zeros(len(batch), 5, dim, dtype=torch.float32)
        return {"q_emb": q_emb, "e_emb": e_emb, "soft": soft, "tool": tools}

    random_split = int(0.9 * len(labels))
    train_recs, val_recs = labels[:random_split], labels[random_split:]

    bs = int(cfg.training.confidence.batch_size)
    lr = float(cfg.training.confidence.learning_rate)
    epochs = int(cfg.training.confidence.epochs)

    train_loader = DataLoader(_Dataset(train_recs), batch_size=bs,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(_Dataset(val_recs), batch_size=bs,
                            shuffle=False, collate_fn=collate)

    head = ConfidenceHead(embedding_dim=encoder.get_sentence_embedding_dimension())
    optim = torch.optim.Adam(head.parameters(), lr=lr)

    for epoch in range(epochs):
        head.train()
        tot = 0.0
        n = 0
        for batch in train_loader:
            out = head(batch["q_emb"], batch["e_emb"])
            conf = out["confidence"].squeeze(-1)
            tool_logits = out["tool_logits"]
            loss_conf = F.mse_loss(conf, batch["soft"])
            loss_tool = F.cross_entropy(tool_logits, batch["tool"])
            loss = loss_conf + 0.5 * loss_tool
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item()
            n += 1
        logger.info("Epoch %d: loss=%.4f", epoch + 1, tot / max(n, 1))

    # Temperature scaling on validation.
    head.eval()
    all_conf = []
    all_soft = []
    with torch.no_grad():
        for batch in val_loader:
            out = head(batch["q_emb"], batch["e_emb"])
            all_conf.extend(out["confidence"].squeeze(-1).cpu().tolist())
            all_soft.extend(batch["soft"].cpu().tolist())
    arr_conf = np.asarray(all_conf)
    arr_soft = np.asarray(all_soft)
    # Grid-search T in [0.5, 3.0] minimising MSE of scaled confidences.
    best_T, best_mse = 1.0, float("inf")
    for T in np.linspace(0.5, 3.0, 26):
        logits = np.log(arr_conf.clip(1e-6, 1 - 1e-6)
                         / (1 - arr_conf).clip(1e-6, 1 - 1e-6))
        scaled = 1.0 / (1.0 + np.exp(-logits / T))
        mse = float(np.mean((scaled - arr_soft) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_T = float(T)
    head.temperature = best_T
    ece = compute_ece(arr_conf, (arr_soft > 0.5).astype(float))

    out_dir = Path(cfg.checkpoints.confidence_head)
    out_dir.mkdir(parents=True, exist_ok=True)
    head.save(out_dir)
    logger.info("Saved confidence head to %s (T=%.3f, ECE=%.4f)",
                out_dir, best_T, ece)
    return {"status": "ok", "temperature": best_T, "ece": ece,
            "output_dir": str(out_dir)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()
