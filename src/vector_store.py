"""
Vector Store Module
Manages the FAISS index for fast similarity search over paper embeddings.
"""

import os
import numpy as np
import pandas as pd
import faiss
import pickle
import logging

logger = logging.getLogger(__name__)

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "faiss.index")
META_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "metadata.pkl")

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP (Inner Product) index.
    Because embeddings are L2-normalised, inner product == cosine similarity.
    """

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []  # parallel list to FAISS vectors

    # ------------------------------------------------------------------
    # Building / Persisting
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Build a new FAISS index from scratch.

        Args:
            embeddings: Float32 array of shape (N, 384).
            metadata:   List of dicts with paper info, length N.
        """
        assert embeddings.shape[1] == EMBEDDING_DIM, (
            f"Expected embedding dim {EMBEDDING_DIM}, got {embeddings.shape[1]}"
        )
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)
        self.metadata = metadata
        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def save(
        self,
        index_path: str = INDEX_PATH,
        meta_path: str = META_PATH,
    ) -> None:
        """Persist the FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Index saved to '{index_path}' ({self.index.ntotal} vectors).")

    def load(
        self,
        index_path: str = INDEX_PATH,
        meta_path: str = META_PATH,
    ) -> bool:
        """
        Load a previously saved FAISS index and metadata.

        Returns:
            True if loaded successfully, False if files do not exist.
        """
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.warning("FAISS index files not found. Build the index first.")
            return False

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors.")
        return True

    def is_ready(self) -> bool:
        """Return True if the index is built and non-empty."""
        return self.index is not None and self.index.ntotal > 0

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        exclude_first: bool = False,
    ) -> list[dict]:
        """
        Search for the top-k most similar papers.

        Args:
            query_embedding: 1D float32 array of shape (384,).
            top_k:           Number of results to return.
            exclude_first:   If True, skip the closest match (useful when the
                             uploaded paper is already in the index).

        Returns:
            List of result dicts, each containing paper metadata +
            'similarity_score' (0–1) and 'similarity_pct' (e.g. "87.4%").
        """
        if not self.is_ready():
            raise RuntimeError("FAISS index is not loaded. Call build() or load() first.")

        query = query_embedding.reshape(1, -1).astype(np.float32)

        fetch_k = top_k + (1 if exclude_first else 0)
        distances, indices = self.index.search(query, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = dict(self.metadata[idx])
            # Clamp cosine similarity to [0, 1] (rounding errors can give slightly >1)
            score = float(min(max(dist, 0.0), 1.0))
            meta["similarity_score"] = round(score, 4)
            meta["similarity_pct"] = f"{score * 100:.1f}%"
            results.append(meta)

        if exclude_first and results:
            results = results[1:]

        return results[:top_k]

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Add new embeddings to an existing index.

        Args:
            embeddings: Float32 array of shape (M, 384).
            metadata:   List of M dicts.
        """
        if not self.is_ready():
            self.build(embeddings, metadata)
            return
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        logger.info(f"Added {len(metadata)} vectors. Total: {self.index.ntotal}")


# ---------------------------------------------------------------------------
# Module-level singleton for reuse across Streamlit reruns
# ---------------------------------------------------------------------------
_store: VectorStore | None = None


def get_store() -> VectorStore:
    """Return the module-level VectorStore singleton."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
