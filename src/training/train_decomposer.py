"""
Train the multi-part query DecompositionClassifier (binary).

Also serialises the 3-shot splitter prompt used at inference time so the
runtime :class:`QuerySplitter` can load it from disk without hardcoding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def train(cfg: Any = None) -> dict[str, Any]:
    cfg = cfg if cfg is not None else get_config()
    set_seed(42)

    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
    except ImportError as exc:
        logger.error("torch required: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    from src.decomposer.classifier import DecompositionClassifier
    from src.retrieval.vector_store import ChromaVectorStore

    labels = _load_jsonl(Path(cfg.data.synthetic.decomp_labels_path))
    if not labels:
        # Even if no training data, save the splitter prompt.
        _save_prompt(Path(cfg.checkpoints.decomposer))
        return {"status": "skipped", "reason": "no_labels_prompt_saved"}

    encoder = ChromaVectorStore().model
    dim = encoder.get_sentence_embedding_dimension()
    clf = DecompositionClassifier(embedding_dim=dim)

    class _Dataset(Dataset):
        def __init__(self, recs: list[dict[str, Any]]):
            self.recs = recs

        def __len__(self) -> int:
            return len(self.recs)

        def __getitem__(self, i: int) -> dict[str, Any]:
            return self.recs[i]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        queries = [b["query"] for b in batch]
        lbl = torch.tensor(
            [1.0 if b.get("is_multi_part") else 0.0 for b in batch],
            dtype=torch.float32,
        )
        emb = torch.tensor(
            encoder.encode(queries, normalize_embeddings=True),
            dtype=torch.float32,
        )
        return {"emb": emb, "label": lbl}

    bs = int(cfg.training.decomposer.batch_size)
    lr = float(cfg.training.decomposer.learning_rate)
    epochs = int(cfg.training.decomposer.epochs)

    split = int(0.9 * len(labels))
    tr, va = labels[:split], labels[split:]
    loader = DataLoader(_Dataset(tr), batch_size=bs, shuffle=True,
                        collate_fn=collate)
    optim = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(epochs):
        clf.train()
        tot, n = 0.0, 0
        for batch in loader:
            logits = clf.forward(batch["emb"]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item()
            n += 1
        logger.info("Epoch %d: loss=%.4f", epoch + 1, tot / max(n, 1))

    out_dir = Path(cfg.checkpoints.decomposer)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save(out_dir)
    _save_prompt(out_dir)
    logger.info("Saved decomposer to %s", out_dir)
    return {"status": "ok", "output_dir": str(out_dir)}


def _save_prompt(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splitter_prompt.txt").write_text(_SPLITTER_PROMPT, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()
