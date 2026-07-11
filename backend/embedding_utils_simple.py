import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

_MODEL = None


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


async def generate_embeddings(chunks: List[str], for_query: bool = False):
    """Generate embeddings for text chunks using Gemini's gemini-embedding-2 if available, otherwise falling back to sentence-transformers."""
    if not chunks:
        return np.empty((0, 3072), dtype=np.float32)

    import os
    import time
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        task_type = "retrieval_query" if for_query else "retrieval_document"
        
        # Retry parameters for rate limits (429)
        max_retries = 3
        delay = 1.0
        backoff_factor = 2.0
        
        for attempt in range(max_retries):
            try:
                res = genai.embed_content(
                    model="models/gemini-embedding-2",
                    content=chunks,
                    task_type=task_type
                )
                embeddings = res.get("embedding", [])
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error: Gemini embedding generation failed after {max_retries} attempts ({e}). Falling back to local SentenceTransformer (Note: this may cause a ChromaDB dimension mismatch error if using a 3072-dim collection).")
                    break # Break out of loop to fall through to local model fallback
                
                # Parse rate-limit wait time dynamically from the API error message
                error_str = str(e)
                sleep_time = delay
                match = re.search(r"Please retry in (\d+\.?\d*)s", error_str)
                if match:
                    sleep_time = float(match.group(1)) + 1.0 # Add 1s buffer
                
                import asyncio
                print(f"Warning: Gemini embedding failed. Retrying in {sleep_time}s...")
                await asyncio.sleep(sleep_time)
                delay *= backoff_factor

    # Local sentence-transformers fallback (used only if API Key is not configured)
    # Ensure every chunk has at least some placeholder text if it's empty to preserve list length
    cleaned_chunks = []
    for chunk in chunks:
        cleaned = preprocess_text(chunk)
        cleaned_chunks.append(cleaned if cleaned else "[empty]")

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
