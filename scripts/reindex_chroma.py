"""
Re-index ChromaDB with a fine-tuned retriever model.

Must be run after training the retriever so that query embeddings and indexed
vectors are in the same embedding space.

Usage
-----
    python scripts/reindex_chroma.py --model checkpoints/retriever

What it does
------------
1. Loads all chunks from the existing ChromaDB collection.
2. Re-embeds them using the fine-tuned SentenceTransformer model.
3. Writes a NEW collection ``aegis_chunks_ft`` (keeping the original intact).
4. Prints instructions for switching the pipeline to the new collection.

The original ``aegis_chunks`` collection is NOT deleted so you can compare
base vs. fine-tuned recall before committing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/retriever",
                        help="Path to fine-tuned SentenceTransformer checkpoint.")
    parser.add_argument("--src-collection", default="aegis_chunks",
                        help="Source ChromaDB collection to re-embed.")
    parser.add_argument("--dst-collection", default="aegis_chunks_ft",
                        help="Destination collection name for fine-tuned embeddings.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from src.utils.config import get_config
    cfg = get_config()

    # ------------------------------------------------------------------
    # Load the fine-tuned model
    # ------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model checkpoint not found: %s", model_path)
        return
    logger.info("Loading fine-tuned model from %s ...", model_path)
    model = SentenceTransformer(str(model_path))

    # ------------------------------------------------------------------
    # Read all chunks from the source collection
    # ------------------------------------------------------------------
    import chromadb
    db_path = cfg.resolve_path(cfg.data.vector_db_path)
    client = chromadb.PersistentClient(path=db_path)

    src = client.get_collection(args.src_collection)
    total = src.count()
    logger.info("Source collection '%s' has %d chunks. Re-embedding ...", args.src_collection, total)

    # Fetch in pages to avoid OOM
    PAGE = 1000
    all_ids, all_docs, all_metas = [], [], []
    offset = 0
    while offset < total:
        page = src.get(limit=PAGE, offset=offset, include=["documents", "metadatas"])
        all_ids.extend(page["ids"])
        all_docs.extend(page["documents"] or [""] * len(page["ids"]))
        all_metas.extend(page["metadatas"] or [{}] * len(page["ids"]))
        offset += PAGE
        logger.info("  Fetched %d / %d chunks ...", min(offset, total), total)

    # ------------------------------------------------------------------
    # Re-embed and write to destination collection
    # ------------------------------------------------------------------
    # Delete existing dst collection if present
    try:
        client.delete_collection(args.dst_collection)
        logger.info("Deleted existing collection '%s'.", args.dst_collection)
    except Exception:
        pass

    dst = client.create_collection(
        args.dst_collection,
        metadata={"hnsw:space": "cosine"},
    )

    bs = args.batch_size
    for i in range(0, len(all_ids), bs):
        batch_ids   = all_ids[i:i + bs]
        batch_docs  = all_docs[i:i + bs]
        batch_metas = all_metas[i:i + bs]

        embeddings = model.encode(
            batch_docs,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        dst.add(ids=batch_ids, documents=batch_docs,
                metadatas=batch_metas, embeddings=embeddings)

        if (i // bs) % 10 == 0:
            logger.info("  Indexed %d / %d chunks ...", min(i + bs, len(all_ids)), len(all_ids))

    logger.info("Re-indexing complete. Collection '%s' has %d chunks.", args.dst_collection, dst.count())

    # ------------------------------------------------------------------
    # Instructions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Re-indexing complete!")
    print(f"  Source collection : {args.src_collection}  ({total} chunks, base embeddings)")
    print(f"  New collection    : {args.dst_collection}  ({dst.count()} chunks, fine-tuned embeddings)")
    print()
    print("To use the fine-tuned retriever, update config/base.yaml:")
    print("  data:")
    print(f"    vector_db_collection: {args.dst_collection}")
    print("  checkpoints:")
    print("    use_finetuned_retriever: true")
    print()
    print("Then run eval to compare recall vs. the base collection.")
    print("=" * 60)


if __name__ == "__main__":
    main()
