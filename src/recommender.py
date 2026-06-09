"""
Recommender Module
Orchestrates the full recommendation pipeline:
  PDF → text → embedding → FAISS search → recommendations
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from collections import Counter

from src.text_processor import (
    extract_text_from_bytes,
    extract_text_from_pdf,
    preprocess_for_embedding,
)
from src.embedder import generate_embedding, generate_embeddings_batch
from src.vector_store import VectorStore, get_store

logger = logging.getLogger(__name__)

# Paths
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "papers.csv")


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index_from_csv(csv_path: str = DATA_PATH) -> VectorStore:
    """
    Load papers CSV, generate embeddings, build FAISS index, and save it.

    Returns:
        Populated VectorStore instance.
    """
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)

    logger.info(f"Building index from {len(df)} papers…")

    # Combine title + abstract for richer embeddings
    texts = (df["title"] + ". " + df["abstract"]).tolist()
    clean_texts = [preprocess_for_embedding(t, use_abstract_only=False) for t in texts]

    embeddings = generate_embeddings_batch(clean_texts, show_progress=True)

    metadata = df.to_dict(orient="records")

    store = get_store()
    store.build(embeddings, metadata)
    store.save()

    return store


def load_or_build_index(csv_path: str = DATA_PATH) -> VectorStore:
    """
    Load existing FAISS index if available; otherwise build from CSV.

    Returns:
        Ready VectorStore instance.
    """
    store = get_store()
    if store.is_ready():
        return store

    loaded = store.load()
    if loaded:
        return store

    if os.path.exists(csv_path):
        return build_index_from_csv(csv_path)

    logger.error("No data available. Fetch papers first.")
    return store


# ---------------------------------------------------------------------------
# Core recommendation
# ---------------------------------------------------------------------------

def recommend_from_bytes(
    pdf_bytes: bytes,
    top_k: int = 5,
) -> dict:
    """
    Full pipeline for an uploaded PDF.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.
        top_k:     Number of similar papers to return.

    Returns:
        Dict with keys:
          'extracted_text'   – raw text from PDF
          'abstract_snippet' – cleaned abstract used for embedding
          'similar_papers'   – list of top-k result dicts
          'authors'          – recommended authors list
          'directions'       – suggested research directions
          'keywords'         – extracted keywords
    """
    store = load_or_build_index()
    if not store.is_ready():
        return {"error": "Index not ready. Please build the dataset first."}

    # Step 1: Extract text
    raw_text = extract_text_from_bytes(pdf_bytes)
    if not raw_text.strip():
        return {"error": "Could not extract text from the uploaded PDF."}

    # Step 2: Preprocess
    clean = preprocess_for_embedding(raw_text, use_abstract_only=True)
    if len(clean) < 20:
        clean = preprocess_for_embedding(raw_text, use_abstract_only=False)

    # Step 3: Embed
    embedding = generate_embedding(clean)

    # Step 4: Search
    similar_papers = store.search(embedding, top_k=top_k, exclude_first=False)

    # Step 5: Derive recommendations
    authors = extract_author_recommendations(similar_papers)
    directions = suggest_research_directions(similar_papers)
    keywords = extract_keywords(clean)

    return {
        "extracted_text": raw_text[:3000],   # truncate for display
        "abstract_snippet": clean[:800],
        "similar_papers": similar_papers,
        "authors": authors,
        "directions": directions,
        "keywords": keywords,
    }


def recommend_from_text(
    query_text: str,
    top_k: int = 5,
) -> dict:
    """
    Recommend papers from a free-text query (title or abstract text).
    """
    store = load_or_build_index()
    if not store.is_ready():
        return {"error": "Index not ready."}

    clean = preprocess_for_embedding(query_text, use_abstract_only=False)
    embedding = generate_embedding(clean)
    similar_papers = store.search(embedding, top_k=top_k)

    return {
        "abstract_snippet": clean[:800],
        "similar_papers": similar_papers,
        "authors": extract_author_recommendations(similar_papers),
        "directions": suggest_research_directions(similar_papers),
        "keywords": extract_keywords(clean),
    }


# ---------------------------------------------------------------------------
# Auxiliary helpers
# ---------------------------------------------------------------------------

def extract_author_recommendations(similar_papers: list[dict], top_n: int = 8) -> list[str]:
    """
    Extract the most frequent / relevant authors from the similar papers.
    """
    author_counts: Counter = Counter()
    for paper in similar_papers:
        raw = paper.get("authors", "")
        for author in raw.split(","):
            author = author.strip()
            if author:
                author_counts[author] += 1

    return [author for author, _ in author_counts.most_common(top_n)]


_DIRECTION_KEYWORDS = {
    "drone": [
        "Federated Learning for Drones",
        "Lightweight Authentication for IoD",
        "Blockchain-based Drone Security",
        "PUF-based Authentication",
        "Edge Computing for UAVs",
    ],
    "security": [
        "Zero-Trust Security Architectures",
        "Post-Quantum Cryptography",
        "Explainable Intrusion Detection",
        "AI-powered Threat Intelligence",
        "Homomorphic Encryption",
    ],
    "nlp": [
        "Large Language Models for Domain Adaptation",
        "Retrieval-Augmented Generation (RAG)",
        "Cross-lingual Transfer Learning",
        "Prompt Engineering Techniques",
        "Multimodal NLP Systems",
    ],
    "vision": [
        "Vision-Language Foundation Models",
        "3D Scene Understanding",
        "Few-Shot Object Detection",
        "Neural Radiance Fields (NeRF)",
        "Diffusion Models for Image Synthesis",
    ],
    "federated": [
        "Privacy-Preserving Federated Learning",
        "Cross-Device Federated Optimization",
        "Federated Learning with Differential Privacy",
        "Communication-Efficient FL",
        "Personalized Federated Learning",
    ],
    "default": [
        "Transformer-Based Architectures for Scientific Discovery",
        "Self-Supervised Learning",
        "Graph Neural Networks",
        "Continual and Lifelong Learning",
        "Efficient Deep Learning for Edge Devices",
    ],
}


def suggest_research_directions(similar_papers: list[dict], top_n: int = 5) -> list[str]:
    """
    Infer research directions from the categories and titles of similar papers.
    """
    combined = " ".join(
        paper.get("title", "") + " " + paper.get("categories", "")
        for paper in similar_papers
    ).lower()

    for keyword, directions in _DIRECTION_KEYWORDS.items():
        if keyword in combined:
            return directions[:top_n]

    return _DIRECTION_KEYWORDS["default"][:top_n]


def extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """
    Simple frequency-based keyword extraction (stopword-filtered).
    For a lightweight approach without NLTK.
    """
    stopwords = {
        "the", "a", "an", "and", "or", "in", "of", "to", "is", "are",
        "was", "were", "for", "on", "at", "by", "with", "from", "that",
        "this", "it", "be", "as", "we", "our", "has", "have", "been",
        "their", "they", "can", "which", "such", "using", "used", "paper",
        "proposed", "method", "based", "results", "also", "show", "shows",
        "two", "new", "than", "more", "each", "into", "these", "approach",
        "work", "its", "not", "but", "while", "both", "all", "if",
    }
    words = re.findall(r"[a-z]{3,}", text.lower())
    filtered = [w for w in words if w not in stopwords]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]
