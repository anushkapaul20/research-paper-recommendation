"""
Embedder Module
Generates semantic embeddings using Sentence Transformers.
"""

import numpy as np
import logging
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Model choice: fast, lightweight, 384-dim embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_model_instance: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Return a cached SentenceTransformer model instance (singleton).
    Downloads the model on first call.
    """
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading Sentence Transformer model: {MODEL_NAME}")
        _model_instance = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
        logger.info("Model loaded successfully.")
    return _model_instance


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a single 384-dimensional embedding for the given text.

    Args:
        text: Input text string.

    Returns:
        Numpy array of shape (384,).
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype(np.float32)


def generate_embeddings_batch(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts in batches.

    Args:
        texts: List of text strings.
        batch_size: Number of texts per batch.
        show_progress: Show tqdm progress bar.

    Returns:
        Numpy array of shape (N, 384).
    """
    model = get_model()
    logger.info(f"Generating embeddings for {len(texts)} texts...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return embeddings.astype(np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two normalized vectors.
    Since embeddings are L2-normalized, dot product == cosine similarity.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score in [0, 1].
    """
    return float(np.dot(vec_a, vec_b))
