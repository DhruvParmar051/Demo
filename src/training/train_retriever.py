"""
Fine-tune the BGE-m3 retriever with Multiple Negatives Ranking Loss.

For each QA pair we form (query, positive_chunk) and 3 BM25-sampled hard
negatives. Trained with sentence-transformers via manual PyTorch loop to
avoid heavy dependencies.
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


def _load_qa(path: Path) -> list[dict[str, Any]]:
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

    # Build a BM25 index over the positive chunks present in qa.
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
        hits = idx.query(query, top_k=k + 5)
        negatives = [c.text for c, _ in hits if c.chunk_id not in gold_ids][:k]
        rec["hard_negative_texts"] = negatives


def train(cfg: Any = None) -> dict[str, Any]:
    """Train the retriever with MNRL."""
    cfg = cfg if cfg is not None else get_config()
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

    _build_hard_negatives(qa, k=int(cfg.training.retriever.hard_negatives))

    model_name = cfg.models.retriever.name
    logger.info("Loading retriever base %s", model_name)
    model = SentenceTransformer(model_name)

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
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in loader:
            queries = [b["query"] for b in batch]
            positives = [b["positive"] for b in batch]
            q_emb = model.encode(queries, convert_to_tensor=True,
                                  show_progress_bar=False)
            p_emb = model.encode(positives, convert_to_tensor=True,
                                  show_progress_bar=False)

            hn_emb = None
            hn_rows = [b["hard_negatives"] for b in batch]
            if any(hn_rows):
                max_k = max((len(h) for h in hn_rows), default=0)
                if max_k > 0:
                    flat = [h for row in hn_rows for h in row]
                    if flat:
                        flat_emb = model.encode(flat, convert_to_tensor=True,
                                                 show_progress_bar=False)
                        # Pad rows to max_k by repeating last negative.
                        padded = []
                        offset = 0
                        for row in hn_rows:
                            if not row:
                                # No negatives -- repeat the positive to keep shape.
                                padded.append(p_emb[len(padded)].unsqueeze(0)
                                              .repeat(max_k, 1))
                            else:
                                row_emb = flat_emb[offset:offset + len(row)]
                                offset += len(row)
                                if row_emb.size(0) < max_k:
                                    pad = row_emb[-1:].repeat(
                                        max_k - row_emb.size(0), 1
                                    )
                                    row_emb = torch.cat([row_emb, pad], dim=0)
                                padded.append(row_emb)
                        hn_emb = torch.stack(padded, dim=0)

            loss = loss_fn(q_emb, p_emb, hn_emb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / max(n, 1)
        logger.info("Epoch %d: loss=%.4f", epoch + 1, avg)
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
