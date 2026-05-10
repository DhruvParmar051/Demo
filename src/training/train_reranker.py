"""
Fine-tune the cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2).

Training signal
---------------
* **Positives** (label=1): gold (query, cited_chunk) pairs from qa_pairs.jsonl.
* **Hard negatives** (label=0): top-k BM25 results that are NOT the gold chunk.
  These look superficially relevant (same keywords) but are wrong — exactly the
  hard cases a reranker must learn to distinguish from true positives.
* **Random negatives** (label=0): a small fraction of random chunks to keep the
  model calibrated against fully off-topic passages.

Architecture
------------
Uses the SAME model that ColBERTReranker loads at inference time so that the
saved state dict is directly applicable without key remapping.
"""

from __future__ import annotations

import json
import logging
import os
import random
import gc
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.utils.config import get_config
from src.utils.determinism import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# ---------------------------------------------------------------------------
# Hard-negative mining via BM25
# ---------------------------------------------------------------------------

def _mine_hard_negatives(
    qa: list[dict[str, Any]],
    n_hard: int = 5,
    n_random: int = 2,
) -> list[dict[str, Any]]:
    """Build (query, passage, label) triples with BM25 hard negatives.

    For each query:
      - Positives: all cited chunks (label=1).
      - Hard negatives: BM25 top-50 minus gold chunk_ids (label=0).
      - Random negatives: random cited_text from OTHER queries (label=0).
    """
    # Lazy import so the function can be tested without a live BM25 index.
    from src.retrieval.bm25_index import BM25Index

    bm25 = BM25Index()
    bm25_path = Path(get_config().data.bm25_index_path)
    if bm25_path.exists():
        bm25.load(str(bm25_path))
        logger.info("BM25 index loaded (%d docs) for hard-negative mining.", bm25.size)
    else:
        logger.warning("BM25 index not found at %s; falling back to random negatives only.", bm25_path)
        bm25 = None

    # Build a flat pool of (chunk_id, text) for random negatives.
    random_pool: list[tuple[str, str]] = []
    for r in qa:
        for cit in r.get("citations", []):
            t = cit.get("cited_text") or ""
            cid = cit.get("chunk_id") or ""
            if t and cid:
                random_pool.append((cid, t))

    pairs: list[dict[str, Any]] = []

    for r in qa:
        query = r.get("query", "")
        if not query:
            continue

        gold_ids: set[str] = set()
        gold_texts: dict[str, str] = {}

        # --- Positives ---
        for cit in r.get("citations", []):
            t = cit.get("cited_text") or ""
            cid = cit.get("chunk_id") or ""
            if t:
                pairs.append({"query": query, "passage": t, "label": 1.0})
                if cid:
                    gold_ids.add(cid)
                    gold_texts[cid] = t

        if not gold_ids:
            continue

        # --- Hard negatives from BM25 ---
        hard_added = 0
        if bm25 is not None:
            try:
                bm25_results = bm25.query(query_text=query, top_k=50)
                for chunk, _score in bm25_results:
                    if hard_added >= n_hard:
                        break
                    if chunk.chunk_id in gold_ids:
                        continue
                    if not chunk.text:
                        continue
                    pairs.append({"query": query, "passage": chunk.text, "label": 0.0})
                    hard_added += 1
            except Exception as exc:
                logger.debug("BM25 hard-neg mining failed for query: %s", exc)

        # --- Random negatives from other queries' citations ---
        rand_added = 0
        attempts = 0
        while rand_added < n_random and attempts < len(random_pool) * 2:
            attempts += 1
            cid, text = random.choice(random_pool)
            if cid in gold_ids:
                continue
            pairs.append({"query": query, "passage": text, "label": 0.0})
            rand_added += 1

    random.shuffle(pairs)
    pos = sum(1 for p in pairs if p["label"] == 1.0)
    neg = len(pairs) - pos
    logger.info("Built %d pairs: %d positives, %d negatives (hard+random).", len(pairs), pos, neg)
    return pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Any = None) -> dict[str, Any]:
    """Fine-tune the cross-encoder reranker."""
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    cfg = cfg or get_config()
    set_seed(cfg.training.seed)

    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        logger.error("transformers/torch required: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    # ------------------------------------------------------------------
    # Load QA data
    # ------------------------------------------------------------------
    qa_path = Path(cfg.data.synthetic.qa_path)
    if not qa_path.exists():
        logger.error("QA data path does not exist: %s", qa_path)
        return {"status": "skipped", "reason": "no_qa_data"}

    with qa_path.open("r", encoding="utf-8") as f:
        qa = [json.loads(l) for l in f if l.strip()]

    if not qa:
        return {"status": "skipped", "reason": "no_qa_data"}

    logger.info("Loaded %d QA pairs from %s", len(qa), qa_path)

    # ------------------------------------------------------------------
    # Build pairs with hard negatives
    # ------------------------------------------------------------------
    n_hard = int(getattr(cfg.training.reranker, "hard_negatives", 5))
    n_random = int(getattr(cfg.training.reranker, "random_negatives", 2))
    pairs = _mine_hard_negatives(qa, n_hard=n_hard, n_random=n_random)
    if not pairs:
        return {"status": "skipped", "reason": "no_pairs"}

    # Train / val split (90 / 10)
    random.shuffle(pairs)
    split = max(1, int(len(pairs) * 0.9))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    logger.info("Train pairs: %d  Val pairs: %d", len(train_pairs), len(val_pairs))

    # ------------------------------------------------------------------
    # Model — MUST match the model loaded at inference in reranker.py
    # ------------------------------------------------------------------
    name = cfg.models.reranker.name  # cross-encoder/ms-marco-MiniLM-L-6-v2
    logger.info("Loading reranker base model: %s", name)

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model.to(device)
    logger.info("Training on device: %s", device)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    epochs = int(cfg.training.reranker.num_epochs)
    bs = int(cfg.training.reranker.batch_size)
    lr = float(cfg.training.reranker.learning_rate)
    max_len = int(cfg.training.reranker.max_seq_length)
    label_smoothing = float(getattr(cfg.training.reranker, "label_smoothing", 0.1))
    accumulation_steps = max(1, 32 // bs)  # effective batch ≈ 32

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.training.reranker.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    def collate(batch: list[dict[str, Any]]):
        enc = tokenizer(
            [b["query"] for b in batch],
            [b["passage"] for b in batch],
            padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        # Apply label smoothing: 1 → (1 - ls), 0 → ls
        raw = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        labels = raw * (1.0 - label_smoothing) + (1.0 - raw) * label_smoothing
        return {"enc": enc, "labels": labels}

    train_loader = DataLoader(train_pairs, batch_size=bs, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_pairs,   batch_size=bs, shuffle=False, collate_fn=collate)

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0
    out_dir = Path(cfg.checkpoints.reranker)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=True)
        for i, batch in enumerate(pbar):
            enc = {k: v.to(device) for k, v in batch["enc"].items()}
            labels = batch["labels"].to(device)

            logits = model(**enc).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            (loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.reranker.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            del enc, labels, logits, loss

        avg_train = train_loss / max(len(train_loader), 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                enc = {k: v.to(device) for k, v in batch["enc"].items()}
                labels = batch["labels"].to(device)
                logits = model(**enc).logits.squeeze(-1)
                val_loss += F.binary_cross_entropy_with_logits(logits, labels).item()
                del enc, labels, logits

        avg_val = val_loss / max(len(val_loader), 1)
        scheduler.step()

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch + 1, epochs, avg_train, avg_val,
        )

        # Save best checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            # Save only the classifier head weights (same format ColBERTReranker expects)
            torch.save(model.state_dict(), out_dir / "model.pt")
            tokenizer.save_pretrained(out_dir)
            logger.info("  ✓ New best val_loss=%.4f — checkpoint saved.", best_val_loss)
        else:
            patience_counter += 1
            logger.info("  No improvement (%d/%d patience).", patience_counter, patience)
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

    logger.info("Training complete. Best val_loss=%.4f. Saved to %s", best_val_loss, out_dir)
    return {"status": "ok", "best_val_loss": best_val_loss, "output_dir": str(out_dir)}


def main() -> None:
    result = train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
