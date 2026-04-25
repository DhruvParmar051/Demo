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
import gc
import os
from pathlib import Path
from typing import Any
from tqdm import tqdm

from src.utils.config import get_config
from src.utils.determinism import set_seed

# Configure logging for clean terminal output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _build_pairs(qa: list[dict[str, Any]], pos_neg_ratio: float) -> list[dict[str, Any]]:
    """Build list of {query, passage, label} dicts."""
    pairs: list[dict[str, Any]] = []
    all_chunks = []

    # 1. Collect all available text chunks from citations for negatives
    for r in qa:
        for cit in r.get("citations", []):
            text = cit.get("cited_text")
            cid = cit.get("chunk_id")
            if text and cid:
                all_chunks.append((cid, text))

    if not all_chunks:
        logger.error("No text found in 'citations'. Check your JSONL structure.")
        return []

    # 2. Build the actual training pairs
    for r in qa:
        query = r["query"]
        gold_ids = set()
        
        # Get Positives from citations
        positives_found = 0
        for cit in r.get("citations", []):
            text = cit.get("cited_text")
            cid = cit.get("chunk_id")
            if text:
                pairs.append({"query": query, "passage": text, "label": 1.0})
                if cid:
                    gold_ids.add(cid)
                positives_found += 1
        
        # 3. Get Negatives (random chunks that aren't the gold ones)
        if positives_found > 0:
            # pos_neg_ratio = negatives per positive (e.g. 2.0 -> 2 negs per positive).
            n_neg = max(1, int(positives_found * max(pos_neg_ratio, 0.0)))
            for _ in range(n_neg):
                c = random.choice(all_chunks)
                if c[0] in gold_ids:
                    continue
                pairs.append({"query": query, "passage": c[1], "label": 0.0})
                
    random.shuffle(pairs)
    logger.info(f"Built {len(pairs)} pairs for reranker training.")
    return pairs


def train(cfg: Any = None) -> dict[str, Any]:
    """Fine-tune the reranker."""
    # CRITICAL: Allow PyTorch to use all available Unified Memory on Mac
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    cfg = get_config() 
    set_seed(42)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        logger.error("transformers/torch required: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    # Load Data
    qa_path = Path(cfg.data.synthetic.qa_path)
    if not qa_path.exists():
        logger.error("QA data path does not exist: %s", qa_path)
        return {"status": "skipped", "reason": "no_qa_data"}
        
    with qa_path.open("r", encoding="utf-8") as f:
        qa = [json.loads(l) for l in f if l.strip()]
    
    if not qa:
        return {"status": "skipped", "reason": "no_qa_data"}

    # Build Training Pairs
    pos_neg = float(cfg.training.reranker.pos_neg_ratio)
    pairs = _build_pairs(qa, pos_neg)
    if not pairs:
        return {"status": "skipped", "reason": "no_pairs"}

    # Model Setup
    name = cfg.models.reranker.name
    logger.info("Loading reranker with manual wrapper to bypass AutoModel validation: %s", name)
    
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    
    class JinaFlashReranker(nn.Module):
        """Custom wrapper to handle Jina's XLMRobertaFlashConfig."""
        def __init__(self, model_name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

        def forward(self, **kwargs):
            out = self.encoder(**kwargs)
            # Use CLS token (index 0) for classification
            cls_vec = out.last_hidden_state[:, 0, :]
            return self.classifier(cls_vec)

    model = JinaFlashReranker(name)

    device = torch.device("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    ))
    model.to(device)
    logger.info("Reranker training on: %s", device)

    # Hyperparameters
    epochs = int(cfg.training.reranker.epochs)
    bs = int(cfg.training.reranker.batch_size)
    lr = float(cfg.training.reranker.learning_rate)
    max_len = int(cfg.models.reranker.max_seq_length)
    accumulation_steps = 4 

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        queries = [b["query"] for b in batch]
        passages = [b["passage"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        enc = tokenizer(queries, passages, padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
        return {"enc": enc, "labels": labels}

    loader = DataLoader(pairs, batch_size=bs, shuffle=True, collate_fn=collate)

    # Training Loop
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        n = 0
        
        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}", leave=True)
        optim.zero_grad()

        for i, batch in enumerate(pbar):
            enc = {k: v.to(device) for k, v in batch["enc"].items()}
            labels = batch["labels"].to(device)
            
            # Forward pass through custom wrapper
            logits = model(**enc).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            current_loss = loss.item()
            
            # Gradient Accumulation
            (loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            tot += current_loss
            n += 1
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})
            
            del enc, labels, logits, loss

        avg = tot / max(n, 1)
        tqdm.write(f"==> Epoch {epoch + 1} Complete | Avg Loss: {avg:.4f}")
        best = min(best, avg)

    # Save Checkpoint
    out_dir = Path(cfg.checkpoints.reranker)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the internal encoder and the head separately or as a state dict
    torch.save(model.state_dict(), out_dir / "model.pt")
    tokenizer.save_pretrained(out_dir)
    logger.info("Saved reranker state dict to %s", out_dir)

    return {"status": "ok", "best_loss": best, "output_dir": str(out_dir)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()