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

DATA_DIR.mkdir(exist_ok=True)
PAPERS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)

CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(exist_ok=True)

# ChromaDB initialization
chroma_client = chromadb.Client(
    Settings(
        is_persistent=True,
        persist_directory=str(CHROMA_DIR),
    )
)

has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
collection_name = "researchlens_gemini" if has_gemini else "researchlens"
collection = chroma_client.get_or_create_collection(name=collection_name)
