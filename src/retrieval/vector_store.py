"""
AegisRAG - ChromaDB Vector Store

Dense retrieval backed by BGE-m3 embeddings stored in ChromaDB.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Metadata

from src.data.schema import ChunkRecord
from src.utils.config import get_config
from src.utils.device import get_device_string

logger = logging.getLogger(__name__)

# Maximum batch size for ChromaDB upsert (hard limit is ~5461 for Chroma)
_CHROMA_UPSERT_BATCH = 5_000
# Embedding batch size for SentenceTransformer
_EMBED_BATCH = 256


class ChromaVectorStore:
    """Persistent vector store using ChromaDB with BGE-m3 embeddings.

    Parameters
    ----------
    collection_name : str
        Name of the Chroma collection.
    persist_directory : str or None
        Path for ChromaDB on-disk persistence.  Falls back to
        ``cfg.data.vector_db_path`` when *None*.
    embedding_model_name : str
        HuggingFace model id for dense embeddings.
    """

    def __init__(
        self,
        collection_name: str = "aegis_chunks",
        persist_directory: str | None = None,
        embedding_model_name: str = "BAAI/bge-m3",
    ) -> None:
        cfg = get_config()

        if persist_directory is None:
            persist_directory = cfg.data.vector_db_path

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name

        # ----- Embedding model ------------------------------------------------
        device = get_device_string(cfg.device.preferred_device)
        logger.info(
            "Loading embedding model %s on %s", embedding_model_name, device
        )
        self.model = SentenceTransformer(
            embedding_model_name,
            device=device,
            trust_remote_code=True,
        )

        # ----- ChromaDB client ------------------------------------------------
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d documents)",
            collection_name,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Embed and upsert a list of chunks into the collection.

        Embedding is done in batches of ``_EMBED_BATCH`` for memory
        efficiency; ChromaDB upserts are batched at ``_CHROMA_UPSERT_BATCH``.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]

        # Batch-encode --------------------------------------------------------
        logger.info("Encoding %d chunks with %s ...", len(texts), self.embedding_model_name)
        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=_EMBED_BATCH,
            show_progress_bar=len(texts) > _EMBED_BATCH,
            normalize_embeddings=True,
        )

        # Prepare metadata (Chroma only accepts str/int/float/bool values) ----
        metadatas: list[Metadata] = []
        for chunk in chunks:
            meta: dict[str, Any] = {
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "created_at": chunk.created_at,
            }
            if chunk.page_number is not None:
                meta["page_number"] = chunk.page_number
            # Flatten user-supplied metadata (only scalar types)
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[f"meta_{k}"] = v
            metadatas.append(meta)

        # Upsert in batches ---------------------------------------------------
        embeddings_list = embeddings.tolist()
        for start in range(0, len(ids), _CHROMA_UPSERT_BATCH):
            end = start + _CHROMA_UPSERT_BATCH
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings_list[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info("Upserted %d chunks into '%s'", len(ids), self.collection_name)

    def query_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[ChunkRecord, float]]:
        """Retrieve using precomputed embedding (avoids re-encoding query)."""

        if self.collection.count() == 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding.astype(float).tolist()],
            "n_results": min(top_k, self.collection.count() or top_k),
            "include": ["documents", "metadatas", "distances"],
        }

        if filters:
            query_kwargs["where"] = filters

        results = self.collection.query(**query_kwargs)

        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        dists = results.get("distances") or []

        if not ids or not docs or not metas or not dists:
            return []

        chunk_ids = ids[0]
        documents = docs[0]
        distances = dists[0]
        metadatas = cast(list[dict[str, Any]], metas[0])
        
        output: list[tuple[ChunkRecord, float]] = []
        for cid, doc, meta, dist in zip(chunk_ids, documents, metadatas, distances):
            similarity = 1.0 - dist

            user_meta = {}
            for k, v in meta.items():
                if k.startswith("meta_"):
                    user_meta[k[5:]] = v

            chunk = ChunkRecord(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                source=meta.get("source", ""),
                page_number=meta.get("page_number"),
                chunk_index=meta.get("chunk_index", 0),
                token_count=meta.get("token_count", 0),
                metadata=user_meta,
                created_at=meta.get("created_at", ""),
            )

            output.append((chunk, similarity))

        return output

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[ChunkRecord, float]]:
        """Retrieve the *top_k* most similar chunks for *query_text*.

        Parameters
        ----------
        query_text : str
            The user query to embed and search.
        top_k : int
            Number of results to return.
        filters : dict, optional
            ChromaDB ``where`` filter dict, e.g.
            ``{"doc_id": "abc123"}`` or ``{"source": {"$contains": "faq"}}``.

        Returns
        -------
        list of (ChunkRecord, float)
            Chunks paired with cosine similarity scores (higher is better).
        """
        query_embedding = self.model.encode(
            [query_text], normalize_embeddings=True
        ).tolist()

        if self.collection.count() == 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_embeddings": query_embedding,
            "n_results": min(top_k, self.collection.count() or top_k),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_kwargs["where"] = filters

        

        results = self.collection.query(**query_kwargs)

        # Unpack Chroma results (lists of lists) ------------------------------
        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        dists = results.get("distances") or []

        if not ids or not docs or not metas or not dists:
            return []

        chunk_ids = ids[0]
        documents = docs[0]
        distances = dists[0]
        metadatas = cast(list[dict[str, Any]], metas[0])

        output: list[tuple[ChunkRecord, float]] = []
        for cid, doc, meta, dist in zip(chunk_ids, documents, metadatas, distances):
            # Chroma cosine distance = 1 - similarity; convert back
            similarity = 1.0 - dist

            # Reconstruct user metadata
            user_meta = {}
            for k, v in meta.items():
                if k.startswith("meta_"):
                    user_meta[k[5:]] = v

            chunk = ChunkRecord(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                source=meta.get("source", ""),
                page_number=meta.get("page_number"),
                chunk_index=meta.get("chunk_index", 0),
                token_count=meta.get("token_count", 0),
                metadata=user_meta,
                created_at=meta.get("created_at", ""),
            )
            output.append((chunk, similarity))

        return output

    def delete_collection(self) -> None:
        """Delete the entire collection from ChromaDB."""
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'", self.collection_name)
        # Recreate so the object stays usable
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_collection_stats(self) -> dict[str, Any]:
        """Return basic statistics about the collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model_name,
            "document_count": count,
        }
