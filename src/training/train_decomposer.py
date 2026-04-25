"""
AegisRAG: Multi-part Query Decomposition Trainer
------------------------------------------------
This script trains a binary classifier to detect if a query is 'multi-part'
and serializes the few-shot prompt used for LLM-based splitting.
"""

from __future__ import annotations

import json
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader, Dataset

from src.utils.config import get_config
from src.utils.determinism import set_seed
from src.decomposer.classifier import DecompositionClassifier
from src.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# --- Few-Shot Prompt for LLM Runtime ---
_SPLITTER_PROMPT = """You split multi-part customer-support questions into
atomic sub-questions. Reply with a JSON array of strings.

Example 1:
Q: How do I reset my password and update my billing address?
A: ["How do I reset my password?", "How do I update my billing address?"]

Example 2:
Q: What is the refund policy for international orders, and can I also cancel a subscription?
A: ["What is the refund policy for international orders?", "Can I cancel a subscription?"]

Example 3:
Q: Is two-factor authentication mandatory?
A: ["Is two-factor authentication mandatory?"]

Now split:
Q: {query}
A:"""

class ConfigMapper(dict):
    """Allows dot-notation access (e.g., cfg.training.epochs) for standard dicts."""
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                return ConfigMapper(value)
            return value
        except KeyError:
            raise AttributeError(f"Config object has no attribute '{name}'")

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Helper to load synthetic training data."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def _save_prompt(out_dir: Path) -> None:
    """Saves the prompt to disk so QuerySplitter can load it without hardcoding."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "splitter_prompt.txt"
    prompt_path.write_text(_SPLITTER_PROMPT, encoding="utf-8")
    logger.info("Serialized splitter prompt to %s", prompt_path)

def train(cfg: Any = None) -> dict[str, Any]:
    """
    Main training entry point.
    1. Loads embeddings via Chroma's encoder.
    2. Trains a Binary Classifier (multi-part vs. single-part).
    3. Saves model weights and splitting prompt.
    """
    # 1. Initialization
    raw_cfg = cfg if cfg is not None else get_config()
    cfg = ConfigMapper(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    set_seed(42)

    # 2. Path & Data Handling
    labels_path = Path(cfg.data.synthetic.decomp_labels_path)
    out_dir = Path(cfg.checkpoints.decomposer)
    labels = _load_jsonl(labels_path)

    # If no data, we only update the prompt and exit gracefully
    if not labels:
        logger.warning("No synthetic labels found at %s. Saving prompt only.", labels_path)
        _save_prompt(out_dir)
        return {"status": "skipped", "reason": "no_labels_prompt_saved"}

    # 3. Model Setup
    # We use the existing embedding model from Chroma to ensure consistency
    vector_store = ChromaVectorStore()
    encoder = vector_store.model 
    dim = encoder.get_sentence_embedding_dimension()
    
    clf = DecompositionClassifier(embedding_dim=dim)
    
    # Dataset and Collate for embedding queries on the fly
    class DecompDataset(Dataset):
        def __init__(self, recs: list[dict[str, Any]]):
            self.recs = recs
        def __len__(self) -> int: return len(self.recs)
        def __getitem__(self, i: int) -> dict[str, Any]: return self.recs[i]

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        queries = [b["query"] for b in batch]
        labels_tensor = torch.tensor(
            [1.0 if b.get("is_multi_part") else 0.0 for b in batch],
            dtype=torch.float32,
        )
        # Encode text to embeddings
        embeddings = torch.tensor(
            encoder.encode(queries, normalize_embeddings=True),
            dtype=torch.float32,
        )
        return {"emb": embeddings, "label": labels_tensor}

    # 4. Training Parameters
    try:
        bs = int(cfg.training.decomposer.batch_size)
        lr = float(cfg.training.decomposer.learning_rate)
        epochs = int(cfg.training.decomposer.epochs)
    except AttributeError as e:
        logger.error("Missing config keys for decomposer training: %s", e)
        return {"status": "error", "message": "config_incomplete"}

    split_idx = int(0.9 * len(labels))
    train_data = labels[:split_idx]
    loader = DataLoader(DecompDataset(train_data), batch_size=bs, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    # 5. Training Loop
    logger.info("Starting Decomposer training: %d samples, %d epochs", len(train_data), epochs)
    for epoch in range(epochs):
        clf.train()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            logits = clf.forward(batch["emb"]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, avg_loss)

    # 6. Saving Artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "decomposer.pt"
    
    # Using the classifier's internal save method or standard torch save
    if hasattr(clf, 'save'):
        clf.save(str(save_path))
    else:
        torch.save(clf.state_dict(), save_path)
        
    _save_prompt(out_dir)
    
    logger.info("Training complete. Model saved to %s", save_path)
    return {"status": "ok", "checkpoint": str(save_path)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()