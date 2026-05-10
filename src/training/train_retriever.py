"""
Fine-tune the BGE-m3 retriever with Multiple Negatives Ranking Loss.

Training signal
---------------
* **Positives**: the actual cited chunk text from the QA pair citations
  (``cited_text`` field).  NOT the synthesized answer — the retriever must
  learn to surface source chunks, not generated prose.
* **Hard negatives**: BM25 top-k results from the LIVE 21k-doc index that
  are not the gold chunk.  These are the hardest cases the retriever sees at
  inference time.
* **In-batch negatives**: every other positive in the batch acts as a
  negative via MNRL's cross-batch loss, giving up to batch_size-1 extra
  implicit negatives for free.

After training you MUST re-index ChromaDB with the fine-tuned model weights,
otherwise the query embedding space won't match the indexed vectors and recall
will drop.  Run:

    python scripts/reindex_chroma.py --model checkpoints/retriever

Then set ``checkpoints.use_finetuned_retriever: true`` in base.yaml.
"""

from __future__ import annotations

import json
import logging
import random
import gc
import os
from pathlib import Path
from typing import Any
from tqdm import tqdm

from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_qa(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_positive(record: dict[str, Any]) -> str:
    """Return the actual cited chunk text for a QA pair.

    Priority:
    1. ``citations[0].cited_text`` — the literal source chunk stored during
       data generation.  This is what the retriever must learn to surface.
    2. ``gold_chunk_texts[0]`` — alternative field name used by some scripts.
    3. ``answer_with_citations`` — last resort; not ideal but beats empty string.
    """
    for cit in record.get("citations", []):
        t = cit.get("cited_text", "")
        if t and len(t.strip()) > 10:
            return t.strip()
    gct = record.get("gold_chunk_texts", [])
    if gct:
        return gct[0]
    return record.get("answer_with_citations", "")


def _build_hard_negatives(qa: list[dict[str, Any]], k: int = 7) -> None:
    """Attach hard negatives from the LIVE BM25 index to each record.

    Uses the full 21k-doc production index so negatives reflect real retrieval
    difficulty — not a toy mini-index built from only the gold chunks.
    """
    try:
        from src.retrieval.bm25_index import BM25Index
    except ImportError:
        logger.warning("BM25Index not available; skipping hard negatives.")
        return

    cfg = get_config()
    bm25_path = Path(cfg.data.bm25_index_path)
    if not bm25_path.exists():
        logger.warning("BM25 index not found at %s; skipping hard negatives.", bm25_path)
        return

    idx = BM25Index()
    idx.load(str(bm25_path))
    logger.info("BM25 index loaded (%d docs) for hard-negative mining.", idx.size)

    for rec in qa:
        query = rec.get("query", "")
        if not query:
            rec["hard_negative_texts"] = []
            continue
        gold_ids = set(rec.get("gold_chunk_ids", []))
        # Also exclude the chunk_ids from citations to avoid false negatives.
        for cit in rec.get("citations", []):
            cid = cit.get("chunk_id")
            if cid:
                gold_ids.add(cid)

        hits = idx.query(query, top_k=k + 10)
        negatives = [
            chunk.text for chunk, _ in hits
            if chunk.chunk_id not in gold_ids and chunk.text
        ][:k]
        rec["hard_negative_texts"] = negatives


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Any = None) -> dict[str, Any]:
    """Train the retriever with MNRL and hard negatives from the live BM25 index."""
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    cfg = cfg or get_config()
    set_seed(cfg.training.seed)

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        logger.error("sentence-transformers/torch required: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    from src.training.losses.mnrl import MultipleNegativesRankingLoss

    # ------------------------------------------------------------------
    # Load QA data
    # ------------------------------------------------------------------
    qa_path = Path(cfg.data.synthetic.qa_path)
    qa = _load_qa(qa_path)
    if not qa:
        logger.warning("No QA pairs at %s; nothing to train on.", qa_path)
        return {"status": "skipped", "reason": "no_data"}
    logger.info("Loaded %d QA pairs from %s", len(qa), qa_path)

    # ------------------------------------------------------------------
    # Hard negatives from live BM25 index
    # ------------------------------------------------------------------
    hard_neg_count = int(cfg.training.retriever.hard_negatives)
    logger.info("Mining hard negatives (k=%d) from live BM25 index ...", hard_neg_count)
    _build_hard_negatives(qa, k=hard_neg_count)

    # ------------------------------------------------------------------
    # Train / val split (90 / 10)
    # ------------------------------------------------------------------
    random.shuffle(qa)
    split = max(1, int(len(qa) * 0.9))
    train_qa, val_qa = qa[:split], qa[split:]
    logger.info("Train: %d  Val: %d", len(train_qa), len(val_qa))

    # ------------------------------------------------------------------
    # Model — freeze bottom layers so only the top N are trainable.
    # BGE-m3 has 24 transformer layers; training all of them needs ~6 GB
    # for gradients + Adam states alone, which OOMs on Mac.
    # Freezing 20 of 24 layers cuts trainable params by ~85% while still
    # adapting the high-level representations that determine relevance.
    # ------------------------------------------------------------------
    model_name = cfg.models.retriever.name
    logger.info("Loading retriever base model: %s", model_name)
    model = SentenceTransformer(model_name)

    device = (
        "cuda" if __import__("torch").cuda.is_available()
        else ("mps" if __import__("torch").backends.mps.is_available() else "cpu")
    )
    model.to(device)

    # Freeze all parameters first, then unfreeze only the top N layers
    # and the pooling / projection head.
    trainable_layers = int(getattr(cfg.training.retriever, "trainable_top_layers", 4))
    encoder = model[0].auto_model  # XLMRobertaModel inside SentenceTransformer

    # Freeze everything
    for param in encoder.parameters():
        param.requires_grad = False

    # Unfreeze the top N encoder layers
    total_layers = len(encoder.encoder.layer)
    for layer in encoder.encoder.layer[total_layers - trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Always unfreeze the final LayerNorm and pooler
    for param in encoder.pooler.parameters():
        param.requires_grad = True

    # Enable gradient checkpointing to halve activation memory
    try:
        encoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")
    except Exception:
        pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "Training on device: %s | trainable params: %s / %s (top %d layers)",
        device, f"{trainable:,}", f"{total:,}", trainable_layers,
    )

    # ------------------------------------------------------------------
    # Hyperparameters — use small batch + high accumulation to stay in memory
    # ------------------------------------------------------------------
    epochs     = int(cfg.training.retriever.num_epochs)
    batch_size = int(getattr(cfg.training.retriever, "mac_batch_size",
                             cfg.training.retriever.batch_size))
    lr         = float(cfg.training.retriever.learning_rate)
    wd         = float(cfg.training.retriever.weight_decay)
    max_grad   = float(cfg.training.retriever.max_grad_norm)
    # Keep effective batch ≈ 32 regardless of physical batch size
    accum      = max(1, 32 // batch_size)

    class _QADataset(Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            r = self.records[i]
            return {
                "query": r["query"],
                "positive": _extract_positive(r),
                "hard_negatives": r.get("hard_negative_texts", []),
            }

    def collate(xs):
        return xs

    train_loader = DataLoader(_QADataset(train_qa), batch_size=batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(_QADataset(val_qa),   batch_size=batch_size,
                              shuffle=False, collate_fn=collate)

    loss_fn = MultipleNegativesRankingLoss(scale=20.0, similarity="cos")
    optim   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    total_steps  = (len(train_loader) // accum) * epochs
    warmup_steps = max(1, int(total_steps * float(cfg.training.retriever.warmup_ratio)))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps          # linear warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0
    out_dir = Path(cfg.checkpoints.retriever)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n = 0
        optim.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=True)
        for i, batch in enumerate(pbar):
            import torch as _torch
            queries   = [b["query"]    for b in batch]
            positives = [b["positive"] for b in batch]

            fq = model.tokenize(queries)
            fp = model.tokenize(positives)
            for key in fq: fq[key] = fq[key].to(device)
            for key in fp: fp[key] = fp[key].to(device)

            q_emb = model(fq)["sentence_embedding"]
            p_emb = model(fp)["sentence_embedding"]

            # Hard negatives
            hn_emb = None
            hn_rows = [b["hard_negatives"] for b in batch]
            if any(hn_rows):
                max_k = max(len(h) for h in hn_rows)
                if max_k > 0:
                    flat = [t for row in hn_rows for t in row]
                    if flat:
                        fhn = model.tokenize(flat)
                        for key in fhn: fhn[key] = fhn[key].to(device)
                        flat_emb = model(fhn)["sentence_embedding"]
                        padded = []
                        offset = 0
                        for row in hn_rows:
                            if not row:
                                padded.append(p_emb[len(padded)].unsqueeze(0).repeat(max_k, 1))
                            else:
                                row_emb = flat_emb[offset:offset + len(row)]
                                offset += len(row)
                                if row_emb.size(0) < max_k:
                                    pad = row_emb[-1:].repeat(max_k - row_emb.size(0), 1)
                                    row_emb = _torch.cat([row_emb, pad], dim=0)
                                padded.append(row_emb)
                        hn_emb = _torch.stack(padded, dim=0)
                        del fhn, flat_emb

            loss = loss_fn(q_emb, p_emb, hn_emb)
            (loss / accum).backward()

            if (i + 1) % accum == 0:
                _torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optim.step()
                scheduler.step()
                optim.zero_grad()
                if device == "mps":   _torch.mps.empty_cache()
                elif device == "cuda": _torch.cuda.empty_cache()
                gc.collect()

            epoch_loss += loss.item()
            n += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            del q_emb, p_emb, hn_emb, loss, fq, fp

        avg_train = epoch_loss / max(n, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with __import__("torch").no_grad():
            for batch in val_loader:
                queries   = [b["query"]    for b in batch]
                positives = [b["positive"] for b in batch]
                fq = model.tokenize(queries);  [fq.__setitem__(k, fq[k].to(device)) for k in fq]
                fp = model.tokenize(positives); [fp.__setitem__(k, fp[k].to(device)) for k in fp]
                q_emb = model(fq)["sentence_embedding"]
                p_emb = model(fp)["sentence_embedding"]
                val_loss += loss_fn(q_emb, p_emb, None).item()

        avg_val = val_loss / max(len(val_loader), 1)
        logger.info("Epoch %d/%d | train=%.4f | val=%.4f", epoch+1, epochs, avg_train, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            model.save(str(out_dir))
            logger.info("  ✓ Best val=%.4f — saved to %s", best_val_loss, out_dir)
        else:
            patience_counter += 1
            logger.info("  No improvement (%d/%d patience).", patience_counter, patience)
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

    logger.info("Done. Best val_loss=%.4f. Output: %s", best_val_loss, out_dir)
    logger.warning(
        "IMPORTANT: Re-index ChromaDB with the fine-tuned model before using it:\n"
        "  python scripts/reindex_chroma.py --model %s", out_dir
    )
    return {"status": "ok", "best_val_loss": best_val_loss, "output_dir": str(out_dir)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
