"""
Fine-tune the BGE-m3 retriever with Multiple Negatives Ranking Loss.

For each QA pair we form (query, positive_chunk) and BM25-sampled hard
negatives.

FIX 7: Hard negative count now correctly read from ``cfg.training.retriever.hard_negatives``
       (defaulting to 7, not hard-coded 3).
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

# Configure logging to be clean and non-intrusive for TQDM
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_qa(path: Path) -> list[dict[str, Any]]:
    """Return the raw non-empty lines of a JSONL file."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _build_hard_negatives(qa: list[dict[str, Any]], k: int = 3) -> None:
    """Populate a ``hard_negative_texts`` field on each record via BM25."""
    try:
        from src.retrieval.bm25_index import BM25Index  # type: ignore
    except ImportError:
        return
    from src.data.schema import ChunkRecord

    chunks = []
    seen = set()
    for rec in qa:
        for cid, text in zip(rec.get("gold_chunk_ids", []),
                              rec.get("gold_chunk_texts", [])):
            if cid in seen:
                continue
            seen.add(cid)
            chunks.append(ChunkRecord(
                chunk_id=cid, doc_id=cid[:8], text=text,
                source="", chunk_index=0,
            ))
    if not chunks:
        return
    idx = BM25Index()
    idx.build_index(chunks)

    for rec in qa:
        query = rec["query"]
        gold_ids = set(rec.get("gold_chunk_ids", []))
        # FIX 7: Request k+5 to have enough candidates after filtering gold
        hits = idx.query(query, top_k=k + 5)
        negatives = [c.text for c, _ in hits if c.chunk_id not in gold_ids][:k]
        rec["hard_negative_texts"] = negatives


def train(cfg: Any = None) -> dict[str, Any]:
    """Train the retriever with MNRL."""
    # CRITICAL FIX for Mac OOM: Allow PyTorch to use all available Unified Memory
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    cfg = get_config()
    set_seed(42)

    try:
        import torch  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        logger.error("sentence-transformers/torch required: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    from src.training.losses.mnrl import MultipleNegativesRankingLoss

    qa_path = Path(cfg.data.synthetic.qa_path)
    qa = _load_qa(qa_path)
    if not qa:
        logger.warning("No QA pairs at %s; nothing to train on.", qa_path)
        return {"status": "skipped", "reason": "no_data"}

    # Use config value for hard negatives
    hard_neg_count = int(cfg.training.retriever.hard_negatives)
    logger.info("Building hard negatives with k=%d (from config)", hard_neg_count)
    _build_hard_negatives(qa, k=hard_neg_count)

    model_name = cfg.models.retriever.name
    logger.info("Loading retriever base %s", model_name)
    model = SentenceTransformer(model_name)

    # Hardware-agnostic device selection (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    logger.info("Retriever training running on: %s", device)

    epochs = int(cfg.training.retriever.epochs)
    batch_size = int(cfg.training.retriever.batch_size)
    lr = float(cfg.training.retriever.learning_rate)

    class _QADataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, i: int) -> dict[str, Any]:
            r = self.records[i]
            pos_texts = r.get("gold_chunk_texts") or []
            pos = pos_texts[0] if pos_texts else r.get("answer_with_citations", "")
            return {
                "query": r["query"],
                "positive": pos,
                "hard_negatives": r.get("hard_negative_texts", []),
            }

    dataset = _QADataset(qa)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda xs: xs)

    loss_fn = MultipleNegativesRankingLoss(scale=20.0, similarity="cos")
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    best_loss = float("inf")
    
    # MEMORY SETTINGS: Increase accumulation_steps if OOM persists
    accumulation_steps = 4 
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        
        # Clean UI: Progress bar is the primary feedback source
        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}", leave=True)
        optim.zero_grad()

        for i, batch in enumerate(pbar):
            queries = [b["query"] for b in batch]
            positives = [b["positive"] for b in batch]
            
            # Tokenize and move to device
            features_q = model.tokenize(queries)
            features_p = model.tokenize(positives)
            for k in features_q: features_q[k] = features_q[k].to(device)
            for k in features_p: features_p[k] = features_p[k].to(device)

            # Forward pass (keeping gradients)
            q_emb = model(features_q)['sentence_embedding']
            p_emb = model(features_p)['sentence_embedding']

            # Hard Negatives Processing
            hn_emb = None
            hn_rows = [b["hard_negatives"] for b in batch]
            if any(hn_rows):
                max_k = max((len(h) for h in hn_rows), default=0)
                if max_k > 0:
                    flat = [h for row in hn_rows for h in row]
                    if flat:
                        features_hn = model.tokenize(flat)
                        for k in features_hn: features_hn[k] = features_hn[k].to(device)
                        flat_emb = model(features_hn)['sentence_embedding']
                        
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
                                    row_emb = torch.cat([row_emb, pad], dim=0)
                                padded.append(row_emb)
                        hn_emb = torch.stack(padded, dim=0)

            # --- Backprop with Gradient Accumulation ---
            loss = loss_fn(q_emb, p_emb, hn_emb)
            current_batch_loss = loss.item()
            
            # Scale loss and calculate gradients
            (loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()
                
                # Aggressive Memory Clearing
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # Update Metrics and UI
            epoch_loss += current_batch_loss
            n += 1
            pbar.set_postfix({"loss": f"{current_batch_loss:.4f}"})
            
            # Explicitly delete tensors to assist garbage collector
            del q_emb, p_emb, hn_emb, loss

        avg = epoch_loss / max(n, 1)
        tqdm.write(f"==> Epoch {epoch + 1} Complete | Avg Loss: {avg:.4f}")
        best_loss = min(best_loss, avg)

    out_dir = Path(cfg.checkpoints.retriever)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    logger.info("Saved retriever to %s", out_dir)

    return {"status": "ok", "best_loss": best_loss,
            "output_dir": str(out_dir)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()