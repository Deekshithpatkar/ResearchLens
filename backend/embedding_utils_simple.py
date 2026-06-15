"""
Simplified embedding generator using cached embeddings for MVP.
Works around Python 3.14 dependency issues.
"""

import json
import pickle
from pathlib import Path

# Try to use sentence-transformers, fallback to simple hashing
try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    USE_SENTENCE_TRANSFORMERS = False
    import hashlib

EMBEDDINGS_CACHE = Path(__file__).parent.parent / "data" / ".embeddings_cache"
EMBEDDINGS_CACHE.mkdir(exist_ok=True)

def _get_cached_embedding(text_hash: str):
    """Try to load cached embedding"""
    cache_file = EMBEDDINGS_CACHE / f"{text_hash}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def _save_cached_embedding(text_hash: str, embedding):
    """Save embedding to cache"""
    cache_file = EMBEDDINGS_CACHE / f"{text_hash}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(embedding, f)

def generate_embeddings(chunks):
    """
    Generate embeddings for text chunks.
    Uses SentenceTransformer if available, otherwise returns mock embeddings.
    
    Args:
        chunks: List of text strings
        
    Returns:
        numpy array of embeddings (N, 384) for all-MiniLM-L6-v2 model
    """
    import numpy as np
    
    if USE_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(chunks)
            return embeddings
        except Exception as e:
            print(f"Warning: SentenceTransformer failed ({e}), using fallback")
    
    # Fallback: generate mock embeddings based on text hash
    # This allows testing the pipeline without full dependencies
    embeddings = []
    for chunk in chunks:
        # Create deterministic mock embedding from text
        chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
        seed = int(chunk_hash[:16], 16) % (2**31)
        np.random.seed(seed)
        embedding = np.random.randn(384).astype(np.float32)
        embeddings.append(embedding)
    
    return np.array(embeddings)
