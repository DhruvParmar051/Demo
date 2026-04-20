"""
Fine-tune Jina-ColBERT-v2 as a cross-encoder reranker.

Each training triple is (query, passage, label) where label=1 for gold
(query, positive) pairs and label=0 for random-sample negatives. We use
BCE loss over a single relevance logit.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


def _build_pairs(qa: list[dict[str, Any]], pos_neg_ratio: float) -> list[dict[str, Any]]:
    """Build list of {query, passage, label} dicts."""
    pairs: list[dict[str, Any]] = []
    all_chunks = []
    for r in qa:
        for cid, text in zip(r.get("gold_chunk_ids", []),
                              r.get("gold_chunk_texts", [])):
            all_chunks.append((cid, text))
    seen_ids = {c for c, _ in all_chunks}
    for r in qa:
        gold_ids = set(r.get("gold_chunk_ids", []))
        gold_texts = r.get("gold_chunk_texts", [])
        # Positives
        for text in gold_texts:
            pairs.append({"query": r["query"], "passage": text, "label": 1.0})
        # Negatives
        # pos_neg_ratio = negatives per positive (e.g. 2.0 -> 2 negs per positive).
        n_neg = max(1, int(len(gold_texts) * max(pos_neg_ratio, 0.0)))
        for _ in range(n_neg):
            c = random.choice(all_chunks) if all_chunks else None
            if c is None or c[0] in gold_ids:
                continue
            pairs.append({"query": r["query"], "passage": c[1], "label": 0.0})
    random.shuffle(pairs)
    return pairs


def train(cfg: Any = None) -> dict[str, Any]:
    """Fine-tune the reranker."""
    cfg = cfg if cfg is not None else get_config()
    set_seed(42)

    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        logger.error("transformers/torch required: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    qa_path = Path(cfg.data.synthetic.qa_path)
    if not qa_path.exists():
        return {"status": "skipped", "reason": "no_qa_data"}
    with qa_path.open("r", encoding="utf-8") as f:
        qa = [json.loads(l) for l in f if l.strip()]
    if not qa:
        return {"status": "skipped", "reason": "no_qa_data"}

    pos_neg = float(cfg.training.reranker.pos_neg_ratio)
    pairs = _build_pairs(qa, pos_neg)
    if not pairs:
        return {"status": "skipped", "reason": "no_pairs"}

    name = cfg.models.reranker.name
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=1, trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)

    epochs = int(cfg.training.reranker.epochs)
    bs = int(cfg.training.reranker.batch_size)
    lr = float(cfg.training.reranker.learning_rate)
    max_len = int(cfg.models.reranker.max_seq_length)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        q = [b["query"] for b in batch]
        p = [b["passage"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        enc = tokenizer(q, p, padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
        return {"enc": enc, "labels": labels}

    loader = DataLoader(pairs, batch_size=bs, shuffle=True, collate_fn=collate)

    model.train()
    best = float("inf")
    for epoch in range(epochs):
        tot = 0.0
        n = 0
        for batch in loader:
            enc = {k: v.to(device) for k, v in batch["enc"].items()}
            labels = batch["labels"].to(device)
            out = model(**enc)
            logits = out.logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item()
            n += 1
        avg = tot / max(n, 1)
        logger.info("Epoch %d: loss=%.4f", epoch + 1, avg)
        best = min(best, avg)

    out_dir = Path(cfg.checkpoints.reranker)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Saved reranker to %s", out_dir)

    return {"status": "ok", "best_loss": best, "output_dir": str(out_dir)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()
