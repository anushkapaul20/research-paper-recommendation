"""
Text Processor Module
Handles PDF text extraction and text cleaning/preprocessing.
"""

import re
import string
import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        full_text = "\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from '{pdf_path}'")
        return full_text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return ""


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes (used when handling Streamlit UploadedFile).

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        Extracted text string.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF bytes: {e}")
        return ""


def extract_abstract_section(text: str) -> str:
    """
    Attempt to extract only the Abstract section from paper text.
    Falls back to the first 1000 characters if no abstract is found.

    Args:
        text: Full paper text.

    Returns:
        Abstract text or fallback snippet.
    """
    # Try to find abstract section
    patterns = [
        r"(?i)abstract\s*[\n\r]+([\s\S]{100,1500})(?=\n\s*\n|\nintroduction|\n1\.)",
        r"(?i)abstract[:\.\-–]\s*([\s\S]{100,1500})(?=\n\s*\n|\nintroduction|\n1\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Fallback: first 1000 chars after removing leading whitespace
    cleaned = text.strip()
    return cleaned[:1000] if len(cleaned) > 100 else cleaned


def clean_text(text: str) -> str:
    """
    Clean and normalize text for embedding generation.

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove special characters and punctuation
      4. Collapse whitespace

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)

    # Remove special characters, keep alphanumeric + spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_for_embedding(raw_text: str, use_abstract_only: bool = True) -> str:
    """
    Full pipeline: extract abstract (optional) → clean.

    Args:
        raw_text: Raw text extracted from PDF or API.
        use_abstract_only: If True, try to extract only the abstract section.

    Returns:
        Cleaned text ready for embedding.
    """
    if use_abstract_only:
        text = extract_abstract_section(raw_text)
    else:
        text = raw_text

    return clean_text(text)
