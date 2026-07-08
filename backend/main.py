from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from backend.pdf_utils import extract_text_from_pdf
from backend.text_chunker import chunk_text, preprocess_text
from backend.embedding_utils_simple import generate_embeddings
from backend.llm_utils import generate_response

app = FastAPI(
    title="ResearchLens API",
    description="PDF processing, semantic search, and RAG query engine",
    version="0.2.0"

)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
DATA_DIR = Path(__file__).parent.parent / "data"
METADATA_FILE = DATA_DIR / "metadata.json"
EMBEDDINGS_DIR = DATA_DIR / ".embeddings"
CHUNKS_DIR = DATA_DIR / ".chunks"

DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)

CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(exist_ok=True)

chroma_client = chromadb.Client(
    Settings(
        is_persistent=True,
        persist_directory=str(CHROMA_DIR),
    )
)
collection = chroma_client.get_or_create_collection(name="researchlens")

def load_metadata() -> Dict:
    """Load or create metadata file"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"papers": {}}

def save_metadata(metadata: Dict):
    """Save metadata to file"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def build_summary_text(text: str, max_words: int = 180) -> str:
    """Build a concise summary-like document from the abstract and opening content of a paper."""
    if not text:
        return ""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    abstract_match = re.search(r"abstract\s*(.*?)(?=\s*(?:\d+\s+)?introduction\b)", cleaned, re.I | re.S)
    if abstract_match:
        candidate = abstract_match.group(1).strip()
        if len(candidate.split()) >= 20:
            return candidate[:2000]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    useful_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        lower = sentence.lower()
        if any(token in lower for token in ["google", "brain", "research", "mailto", "www."]):
            continue
        useful_sentences.append(sentence)
        if len(" ".join(useful_sentences).split()) >= max_words:
            break

    summary = " ".join(useful_sentences[:4]).strip()
    if len(summary.split()) < 20:
        summary = cleaned[:1200]
    return summary[:2000]


def expand_query(query: str) -> str:
    """Expand a user query with useful synonyms for paper QA."""
    lowered = query.lower()
    expanded_terms = [query]
    if any(word in lowered for word in ["objective", "objectives", "goal", "goals"]):
        expanded_terms.append("purpose contribution aim")
    if any(word in lowered for word in ["problem", "solve", "solving"]):
        expanded_terms.append("challenge motivation issue")
    if any(word in lowered for word in ["method", "methodology", "approach"]):
        expanded_terms.append("model framework technique")
    if any(word in lowered for word in ["contribution", "contributions", "contribute"]):
        expanded_terms.append("novelty key contribution")
    return " ".join(expanded_terms)


def keyword_overlap_score(query: str, text: str) -> float:
    """Simple token overlap score for hybrid reranking."""
    if not query or not text:
        return 0.0

    stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "this", "that", "of", "to", "and", "for", "in", "on", "with", "from", "paper", "thesis"}

    q_tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if token not in stopwords and len(token) > 2}
    t_tokens = {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stopwords and len(token) > 2}
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / max(1, len(q_tokens))

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": "ResearchLens API v0.1.0",
        "status": "running",
        "endpoints": [
            "POST /upload-pdf/",
            "GET /papers/",
            "GET /dashboard/",
            "POST /search/",
            "POST /query/",
            "GET /papers/{paper_id}"
        ]
    }

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    Extracts text, generates chunks, and creates embeddings.
    """
    try:
        # Save uploaded file
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process PDF
        extracted_text = extract_text_from_pdf(str(file_path))
        if not extracted_text or len(extracted_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        
        chunks = chunk_text(extracted_text)
        summary_text = build_summary_text(extracted_text)
        all_documents = [summary_text] + chunks if summary_text else chunks
        embeddings = generate_embeddings(all_documents)
        
        # Generate paper ID
        paper_id = file.filename.replace(".pdf", "").replace(" ", "_")
        
        # Save chunks and embeddings locally
        chunks_file = CHUNKS_DIR / f"{paper_id}.json"
        with open(chunks_file, 'w') as f:
            json.dump(chunks, f)
        
        embeddings_file = EMBEDDINGS_DIR / f"{paper_id}.npy"
        np.save(embeddings_file, embeddings)

        # Replace any older index entries for this paper before upserting fresh content
        collection.delete(where={"paper_id": paper_id})

        # Save summary and chunk texts to ChromaDB
        ids = [f"{paper_id}-summary"] + [f"{paper_id}-{idx}" for idx in range(len(chunks))]
        metadatas = [
            {
                "paper_id": paper_id,
                "chunk_type": "summary",
                "chunk_index": -1,
                "filename": file.filename,
            }
        ] + [
            {
                "paper_id": paper_id,
                "chunk_type": "chunk",
                "chunk_index": idx,
                "filename": file.filename,
            }
            for idx in range(len(chunks))
        ]
        documents = [summary_text] + chunks if summary_text else chunks
        collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
            embeddings=[emb.tolist() for emb in embeddings],
        )
        
        # Update metadata
        metadata = load_metadata()
        metadata["papers"][paper_id] = {
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "num_chunks": len(chunks),
            "num_indexed_documents": len(documents),
            "text_length": len(extracted_text),
            "embedding_dim": int(embeddings.shape[1])
        }
        save_metadata(metadata)
        
        return {
            "status": "success",
            "paper_id": paper_id,
            "filename": file.filename,
            "total_chunks": len(chunks),
            "total_indexed_documents": len(documents),
            "text_length": len(extracted_text),
            "embedding_dimension": int(embeddings.shape[1]),
            "message": f"Successfully processed {len(chunks)} chunks and {len(documents) - len(chunks)} summary document from {file.filename}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/papers/")
def list_papers():
    """List all processed papers with metadata"""
    metadata = load_metadata()
    return {
        "total_papers": len(metadata.get("papers", {})),
        "papers": metadata.get("papers", {})
    }

@app.get("/dashboard/")
def dashboard():
    """Get current ChromaDB collection status and paper metadata"""
    metadata = load_metadata()
    return {
        "collection_name": collection.name,
        "collection_count": collection.count(),
        "total_papers": len(metadata.get("papers", {})),
        "papers": metadata.get("papers", {}),
        "chroma_directory": str(CHROMA_DIR),
        "stored_paper_ids": list(metadata.get("papers", {}).keys())
    }

@app.get("/papers/{paper_id}")
def get_paper_details(paper_id: str):
    """Get details and chunks for a specific paper"""
    metadata = load_metadata()
    
    if paper_id not in metadata.get("papers", {}):
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")
    
    chunks_file = CHUNKS_DIR / f"{paper_id}.json"
    if not chunks_file.exists():
        raise HTTPException(status_code=404, detail="Chunks file not found")
    
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    return {
        "paper_id": paper_id,
        "metadata": metadata["papers"][paper_id],
        "chunks": chunks
    }

@app.post("/search/")
def semantic_search(query: str, paper_id: str = None, top_k: int = 5):
    """Perform semantic search over paper chunks and prioritize title/summary-like context."""
    try:
        if not query or len(query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        cleaned_query = preprocess_text(query)
        expanded_query = expand_query(cleaned_query)
        query_vectors = generate_embeddings([cleaned_query, expanded_query], for_query=True)
        query_embedding = np.mean(query_vectors, axis=0)

        metadata = load_metadata()
        papers_to_search = {paper_id: metadata["papers"][paper_id]} if paper_id else metadata.get("papers", {})

        if paper_id and paper_id not in papers_to_search:
            raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

        query_kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": max(top_k * 5, 12),
            "include": ["metadatas", "documents", "distances"],
        }
        if paper_id:
            query_kwargs["where"] = {"paper_id": paper_id}

        query_results = collection.query(**query_kwargs)

        if not query_results["ids"] or len(query_results["ids"][0]) == 0:
            return {
                "query": query,
                "results": [],
                "message": "No chunks found"
            }

        ranked_results = []
        for doc, meta, distance in zip(
            query_results["documents"][0],
            query_results["metadatas"][0],
            query_results["distances"][0],
        ):
            cleaned_doc = preprocess_text(doc)
            text = cleaned_doc or doc
            lowered = text.lower()
            boost = 0.0

            if meta.get("chunk_type") == "summary":
                boost += 0.12
            if any(word in lowered for word in ["abstract", "objective", "goal", "purpose", "problem", "contribution", "introduc", "method", "results"]):
                boost += 0.06
            if any(word in lowered for word in ["dataset", "model", "architecture", "attention", "transformer"]):
                boost += 0.01
            if len(text.split()) < 40:
                boost -= 0.02
            if len(text.split()) > 80:
                boost += 0.02

            keyword_score = keyword_overlap_score(cleaned_query, text)
            similarity = float(1.0 / (1.0 + distance)) + boost + (0.05 * keyword_score)
            ranked_results.append({
                "paper_id": meta.get("paper_id"),
                "chunk_index": meta.get("chunk_index"),
                "chunk_type": meta.get("chunk_type", "chunk"),
                "chunk_preview": text[:220] + "..." if len(text) > 220 else text,
                "similarity_score": similarity,
            })

        ranked_results = sorted(ranked_results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]

        return {
            "query": query,
            "num_results": len(ranked_results),
            "results": ranked_results,
            "retrieval_note": "Results are ranked from summary-style context and detailed chunks with lightweight keyword reranking"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.post("/query/")
def query_papers(query: str, paper_id: str = None, top_k: int = 5):
    """
    Perform semantic search and generate a synthesized, polished response using Gemini.
    """
    try:
        search_results = semantic_search(query=query, paper_id=paper_id, top_k=top_k)
        chunks = search_results.get("results", [])
        
        answer = generate_response(query=query, chunks=chunks)
        
        return {
            "query": query,
            "answer": answer,
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating RAG response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)