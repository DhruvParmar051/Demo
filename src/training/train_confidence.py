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


class ConfigMapper(dict):
    """
    Helper class to allow dot-notation access (cfg.key.subkey) 
    on standard Python dictionaries recursively.
    """
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                return ConfigMapper(value)
            return value
        except KeyError:
            raise AttributeError(f"Config object has no attribute '{name}'")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Helper to load JSONL records from a path."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def train(cfg: Any = None) -> dict[str, Any]:
    """
    Main training routine for the Confidence and Tool Routing heads.
    """
    # 1. Initialize configuration and wrap for dot-access
    raw_cfg = cfg if cfg is not None else get_config()
    cfg = ConfigMapper(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
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

    # 2. Path resolution: Match base.yaml data structure
    # Fallback to local 'data' if synthetic_dir is missing in cfg
    synth_dir = getattr(cfg.data, 'synthetic_dir', 'data')
    synth_base = Path(synth_dir)
    conf_path = synth_base / "confidence_labels.jsonl"
    tool_path = synth_base / "tool_route_labels.jsonl"

    labels = _load_jsonl(conf_path)
    tool_labels = _load_jsonl(tool_path)
    
    if not labels:
        logger.error(f"Required confidence labels not found at {conf_path}")
        return {"status": "skipped", "reason": "no_labels"}

    # Map queries to their gold tools
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

        # Gold tool indices — default to SearchKB (not AnswerDirect) when the
        # query has no explicit label, to avoid collapsing the tool head.
        def _safe_tool(q: str) -> int:
            t = tool_lookup.get(q, "SearchKB")
            return _TOOL_NAMES.index(t) if t in _TOOL_NAMES else 1  # 1=SearchKB
        tools = torch.tensor([_safe_tool(b["query"]) for b in batch], dtype=torch.long)

        q_emb = torch.tensor(
            encoder.encode(queries, normalize_embeddings=True),
            dtype=torch.float32,
        )

        # Retrieve top-5 evidence chunks per query and mean-pool their embeddings.
        # Previously this was always zero, which caused the confidence head to
        # ignore evidence entirely and collapse the tool policy to AnswerDirect.
        dim = q_emb.size(-1)
        e_rows = []
        for i, b in enumerate(batch):
            try:
                results = vstore.query_by_embedding(
                    embedding=q_emb[i].numpy(), top_k=5
                )
                if results:
                    texts = [r[0].text for r in results]
                    embs = encoder.encode(texts, normalize_embeddings=True)
                    e_rows.append(torch.tensor(embs, dtype=torch.float32).mean(0))
                    continue
            except Exception:
                pass
            e_rows.append(torch.zeros(dim, dtype=torch.float32))
        e_emb = torch.stack(e_rows)

        return {"q_emb": q_emb, "e_emb": e_emb, "soft": soft, "tool": tools}

    # 3. Data split and Hyperparameters
    random_split = int(0.9 * len(labels))
    train_recs, val_recs = labels[:random_split], labels[random_split:]

    train_params = cfg.training.confidence_head
    bs = int(train_params.batch_size)
    lr = float(train_params.learning_rate)
    epochs = int(train_params.num_epochs)

    train_loader = DataLoader(_Dataset(train_recs), batch_size=bs,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(_Dataset(val_recs), batch_size=bs,
                            shuffle=False, collate_fn=collate)

    # 4. Model Setup
    head = ConfidenceHead(embedding_dim=encoder.get_sentence_embedding_dimension())
    optim = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=float(train_params.weight_decay))

    # 5. Training Loop
    for epoch in range(epochs):
        head.train()
        tot, n = 0.0, 0
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
        logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, epochs, tot / max(n, 1))

    # 6. Temperature scaling on validation
    head.eval()
    all_conf = []
    all_soft = []
    with torch.no_grad():
        for batch in val_loader:
            out = head(batch["q_emb"], batch["e_emb"])
            
            # FIX: Ensure tensors are at least 1D for list conversion
            conf_vals = out["confidence"].squeeze(-1).view(-1).cpu().tolist()
            soft_vals = batch["soft"].view(-1).cpu().tolist()
            
            all_conf.extend(conf_vals)
            all_soft.extend(soft_vals)
            
    arr_conf = np.asarray(all_conf)
    arr_soft = np.asarray(all_soft)
    
    best_T, best_mse = 1.0, float("inf")
    for T in np.linspace(0.5, 3.0, 26):
        logits = np.log(arr_conf.clip(1e-6, 1 - 1e-6)
                         / (1 - arr_conf).clip(1e-6, 1 - 1e-6))
        scaled = 1.0 / (1.0 + np.exp(-logits / T))
        mse = float(np.mean((scaled - arr_soft) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_T = float(T)
            
    # FIX: Assign temperature as a torch.Tensor to satisfy PyTorch buffer requirements
    head.temperature = torch.tensor(best_T)
    ece = compute_ece(arr_conf, (arr_soft > 0.5).astype(float))

    # 7. Save results
    out_dir = Path(train_params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # FIX: Provide a full file path instead of just a directory to head.save
    save_path = out_dir / "confidence_head.pt"
    head.save(save_path)
    
    logger.info("Saved confidence head to %s (T=%.3f, ECE=%.4f)",
                save_path, best_T, ece)
    
    return {
        "status": "ok", 
        "temperature": float(best_T), 
        "ece": ece,
        "output_dir": str(save_path)
    }


def main() -> None:
    """CLI entry point for training."""
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()