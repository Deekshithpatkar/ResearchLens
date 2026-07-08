"""Embedding utilities with lightweight cleaning and a reliable lexical-semantic fallback."""

import hashlib
import pickle
import re
from pathlib import Path
from typing import List

import numpy as np

from backend.text_chunker import preprocess_text

try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    USE_SENTENCE_TRANSFORMERS = False

EMBEDDINGS_CACHE = Path(__file__).parent.parent / "data" / ".embeddings_cache"
EMBEDDINGS_CACHE.mkdir(exist_ok=True)

_MODEL = None


def _get_cached_embedding(text_hash: str):
    cache_file = EMBEDDINGS_CACHE / f"{text_hash}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def _save_cached_embedding(text_hash: str, embedding):
    cache_file = EMBEDDINGS_CACHE / f"{text_hash}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(embedding, f)


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not USE_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers is not available")

    for model_name in ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]:
        try:
            _MODEL = SentenceTransformer(model_name)
            return _MODEL
        except Exception as exc:
            print(f"Warning: failed to load embedding model {model_name}: {exc}")

    raise RuntimeError("No embedding model could be loaded")


def _tokenize(text: str):
    text = preprocess_text(text).lower()
    return re.findall(r"[a-z0-9]+", text)


def _build_lexical_embedding(text: str, vocab=None, dim=384):
    tokens = _tokenize(text)
    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    if vocab is None:
        vocab = sorted(set(tokens))

    vector = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        idx = abs(hash(token)) % dim
        vector[idx] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def generate_embeddings(chunks: List[str], for_query: bool = False):
    """Generate embeddings for text chunks using Gemini's gemini-embedding-2 if available, otherwise falling back to sentence-transformers."""
    if not chunks:
        return np.empty((0, 3072), dtype=np.float32)

    import os
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            task_type = "retrieval_query" if for_query else "retrieval_document"
            res = genai.embed_content(
                model="models/gemini-embedding-2",
                content=chunks,
                task_type=task_type
            )
            embeddings = res.get("embedding", [])
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Gemini embedding generation failed ({e}), falling back to local model")

    # Local sentence-transformers fallback
    cleaned_chunks = [preprocess_text(chunk) for chunk in chunks]
    cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk]

    if not cleaned_chunks:
        cleaned_chunks = [" ".join(chunks)]

    if USE_SENTENCE_TRANSFORMERS:
        try:
            model = _get_model()
            embeddings = model.encode(
                cleaned_chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            print(f"Warning: SentenceTransformer failed ({exc}), using lexical fallback")

    vocab = set()
    for chunk in cleaned_chunks:
        vocab.update(_tokenize(chunk))
    vocab = sorted(vocab)
    embeddings = [_build_lexical_embedding(chunk, vocab=vocab, dim=384) for chunk in cleaned_chunks]
    return np.array(embeddings, dtype=np.float32)
