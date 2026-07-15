from pathlib import Path
import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Storage paths
DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = Path(__file__).parent.parent / "papers"
METADATA_FILE = DATA_DIR / "metadata.json"
EMBEDDINGS_DIR = DATA_DIR / ".embeddings"
CHUNKS_DIR = DATA_DIR / ".chunks"
PROFILES_DIR = DATA_DIR / "paper_profiles"

DATA_DIR.mkdir(exist_ok=True)
PAPERS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)
PROFILES_DIR.mkdir(exist_ok=True)

# User-specific directory helpers
def get_user_data_dir(user_id: str) -> Path:
    user_dir = DATA_DIR / "users" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def get_user_papers_dir(user_id: str) -> Path:
    d = get_user_data_dir(user_id) / "pdfs"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_user_embeddings_dir(user_id: str) -> Path:
    d = get_user_data_dir(user_id) / "embeddings"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_user_chunks_dir(user_id: str) -> Path:
    d = get_user_data_dir(user_id) / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_user_profiles_dir(user_id: str) -> Path:
    d = get_user_data_dir(user_id) / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_user_timelines_dir(user_id: str) -> Path:
    d = get_user_data_dir(user_id) / "timelines"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_user_clusters_dir(user_id: str) -> Path:
    d = get_user_data_dir(user_id) / "clusters"
    d.mkdir(parents=True, exist_ok=True)
    return d


CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"
SCHEMA_VERSION = "1.1"

import asyncio
_semaphore = None

def get_extraction_semaphore():
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(3)
    return _semaphore

# ChromaDB initialization
chroma_client = chromadb.Client(
    Settings(
        is_persistent=True,
        persist_directory=str(CHROMA_DIR),
    )
)

class DynamicCollectionProxy:
    def __init__(self):
        self._cached_collection = None
        self._cached_has_gemini = None

    @property
    def _collection(self):
        # Dynamically check for key at invocation time and cache resolved collection
        has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
        if self._cached_collection is None or has_gemini != self._cached_has_gemini:
            name = "researchlens_gemini" if has_gemini else "researchlens"
            self._cached_collection = chroma_client.get_or_create_collection(name=name)
            self._cached_has_gemini = has_gemini
        return self._cached_collection

    def __getattr__(self, name):
        return getattr(self._collection, name)

collection = DynamicCollectionProxy()
