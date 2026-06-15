from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np

from backend.pdf_utils import extract_text_from_pdf
from backend.text_chunker import chunk_text
from backend.embedding_utils_simple import generate_embeddings

app = FastAPI(
    title="ResearchLens API",
    description="PDF processing and semantic search engine",
    version="0.1.0"
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

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": "ResearchLens API v0.1.0",
        "status": "running",
        "endpoints": [
            "POST /upload-pdf/",
            "GET /papers/",
            "POST /search/",
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
        embeddings = generate_embeddings(chunks)
        
        # Generate paper ID
        paper_id = file.filename.replace(".pdf", "").replace(" ", "_")
        
        # Save chunks and embeddings
        chunks_file = CHUNKS_DIR / f"{paper_id}.json"
        with open(chunks_file, 'w') as f:
            json.dump(chunks, f)
        
        embeddings_file = EMBEDDINGS_DIR / f"{paper_id}.npy"
        np.save(embeddings_file, embeddings)
        
        # Update metadata
        metadata = load_metadata()
        metadata["papers"][paper_id] = {
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "num_chunks": len(chunks),
            "text_length": len(extracted_text),
            "embedding_dim": int(embeddings.shape[1])
        }
        save_metadata(metadata)
        
        return {
            "status": "success",
            "paper_id": paper_id,
            "filename": file.filename,
            "total_chunks": len(chunks),
            "text_length": len(extracted_text),
            "embedding_dimension": int(embeddings.shape[1]),
            "message": f"Successfully processed {len(chunks)} chunks from {file.filename}"
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
    """
    Perform semantic search across paper chunks.
    If paper_id is provided, search only that paper. Otherwise, search all papers.
    """
    try:
        if not query or len(query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate query embedding
        query_embedding = generate_embeddings([query])[0]
        
        metadata = load_metadata()
        results = []
        
        # Determine which papers to search
        papers_to_search = {paper_id: metadata["papers"][paper_id]} if paper_id else metadata.get("papers", {})
        
        if paper_id and paper_id not in papers_to_search:
            raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")
        
        # Search across papers
        for pid in papers_to_search:
            chunks_file = CHUNKS_DIR / f"{pid}.json"
            embeddings_file = EMBEDDINGS_DIR / f"{pid}.npy"
            
            if not chunks_file.exists() or not embeddings_file.exists():
                continue
            
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
            
            embeddings = np.load(embeddings_file)
            
            # Calculate similarities
            for idx, embedding in enumerate(embeddings):
                similarity = cosine_similarity(query_embedding, embedding)
                results.append({
                    "paper_id": pid,
                    "chunk_id": idx,
                    "chunk_preview": chunks[idx][:200] + "..." if len(chunks[idx]) > 200 else chunks[idx],
                    "similarity_score": float(similarity)
                })
        
        if not results:
            return {
                "query": query,
                "results": [],
                "message": "No chunks found"
            }
        
        # Sort by similarity and return top_k
        results_sorted = sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        
        return {
            "query": query,
            "num_results": len(results_sorted),
            "results": results_sorted
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)