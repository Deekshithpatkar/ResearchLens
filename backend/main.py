import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.config import DATA_DIR, PAPERS_DIR, EMBEDDINGS_DIR, CHUNKS_DIR, collection
from backend.metadata_utils import load_metadata, save_metadata
from backend.search_utils import build_summary_text, execute_semantic_search
from backend.llm_utils import generate_response
from backend.pdf_utils import extract_text_from_pdf
from backend.text_chunker import chunk_text
from backend.embedding_utils_simple import generate_embeddings
from backend.profile_utils import extract_paper_profile
from backend.analytics_utils import execute_global_analytics, execute_cluster_analysis, execute_timeline_generation, execute_hierarchical_clustering

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

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": "ResearchLens API v0.2.0",
        "status": "running",
        "endpoints": [
            "POST /upload-pdf/",
            "GET /papers/",
            "POST /query/",
            "POST /analytics/",
            "POST /analytics/reprocess/",
            "GET /analytics/clusters/cosine/",
            "GET /analytics/clusters/hierarchical/",
            "GET /analytics/timeline/"
        ]
    }

@app.post("/upload-pdf/")
async def upload_pdf(file: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload and process multiple PDF files.
    Extracts text, generates chunks, and creates embeddings for each.
    """
    results = []
    metadata = load_metadata()
    
    for f in file:
        try:
            # Save uploaded file
            file_path = PAPERS_DIR / f.filename
            with open(file_path, "wb") as buffer:
                content = await f.read()
                buffer.write(content)
            
            # Process PDF
            extracted_text = extract_text_from_pdf(str(file_path))
            if not extracted_text or len(extracted_text.strip()) == 0:
                results.append({
                    "filename": f.filename,
                    "status": "error",
                    "detail": "No text extracted from PDF"
                })
                continue
            
            chunks = chunk_text(extracted_text)
            summary_text = build_summary_text(extracted_text)
            all_documents = [summary_text] + chunks if summary_text else chunks
            embeddings = await generate_embeddings(all_documents)
            
            # Generate paper ID
            paper_id = f.filename.replace(".pdf", "").replace(" ", "_")
            
            # Save chunks and embeddings locally
            chunks_file = CHUNKS_DIR / f"{paper_id}.json"
            with open(chunks_file, 'w') as chunks_f:
                json.dump(chunks, chunks_f)
            
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
                    "filename": f.filename,
                }
            ] + [
                {
                    "paper_id": paper_id,
                    "chunk_type": "chunk",
                    "chunk_index": idx,
                    "filename": f.filename,
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
            
            # Update metadata in memory dict
            metadata["papers"][paper_id] = {
                "filename": f.filename,
                "uploaded_at": datetime.now().isoformat(),
                "num_chunks": len(chunks),
                "num_indexed_documents": len(documents),
                "text_length": len(extracted_text),
                "embedding_dim": int(embeddings.shape[1])
            }
            
            results.append({
                "status": "success",
                "paper_id": paper_id,
                "filename": f.filename,
                "total_chunks": len(chunks),
                "total_indexed_documents": len(documents),
                "text_length": len(extracted_text),
                "embedding_dimension": int(embeddings.shape[1]),
                "message": f"Successfully processed {len(chunks)} chunks from {f.filename}"
            })
            
            # Queue background task to extract structured paper profile
            if background_tasks:
                background_tasks.add_task(extract_paper_profile, paper_id, f.filename)
            
        except Exception as e:
            results.append({
                "filename": f.filename,
                "status": "error",
                "detail": str(e)
            })

    # Save the updated metadata once after the loop
    save_metadata(metadata)
    
    # Check if we had any successful uploads
    success_count = sum(1 for r in results if r["status"] == "success")
    if success_count == 0 and len(file) > 0:
        # If all uploads failed, raise an exception or return error detail
        first_error = results[0].get("detail", "Unknown error")
        raise HTTPException(status_code=500, detail=f"All uploads failed. First error: {first_error}")
        
    return {
        "status": "completed",
        "processed_files_count": len(results),
        "results": results
    }

@app.get("/papers/")
def list_papers():
    """List all processed papers with metadata"""
    metadata = load_metadata()
    return {
        "total_papers": len(metadata.get("papers", {})),
        "papers": metadata.get("papers", {})
    }

@app.post("/query/")
async def query_papers(query: str, paper_id: str = None, top_k: int = 8):
    """
    Perform semantic search and generate a synthesized, polished response using Gemini.
    """
    try:
        search_results = await execute_semantic_search(query=query, paper_id=paper_id, top_k=top_k)
        chunks = search_results.get("results", [])
        answer = await generate_response(query=query, chunks=chunks)
        
        return {
            "query": query,
            "answer": answer,
            "chunks": chunks
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating RAG response: {str(e)}")

@app.post("/analytics/")
async def global_analytics(query: str, paper_ids: Optional[str] = Query(None)):
    """
    Perform global, multi-paper analytics comparison (Map-Reduce RAG) over paper profiles using query params.
    Example: ?query=compare objectives&paper_ids=all OR ?query=compare objectives&paper_ids=ALBERT,BERT
    """
    try:
        parsed_ids = None
        if paper_ids:
            parsed_ids = [pid.strip() for pid in paper_ids.split(",") if pid.strip()]
        result = await execute_global_analytics(query=query, paper_ids=parsed_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing analytics: {str(e)}")

@app.post("/analytics/reprocess/")
def reprocess_profiles(background_tasks: BackgroundTasks, paper_ids: Optional[str] = Query(None)):
    """
    Manually re-trigger structured paper profile extraction using query params.
    Example: ?paper_ids=ALBERT,BERT OR empty to reprocess all papers.
    """
    metadata = load_metadata()
    all_paper_ids = list(metadata.get("papers", {}).keys())

    target_ids = all_paper_ids
    if paper_ids:
        target_ids = [pid.strip() for pid in paper_ids.split(",") if pid.strip()]
        
    target_ids = [pid for pid in target_ids if pid in all_paper_ids]

    triggered = []
    for pid in target_ids:
        filename = metadata["papers"][pid].get("filename", f"{pid}.pdf")
        background_tasks.add_task(extract_paper_profile, pid, filename)
        triggered.append(pid)

    return {
        "status": "triggered",
        "message": f"Queued profile extraction background tasks for {len(triggered)} papers.",
        "papers": triggered
    }

@app.get("/analytics/clusters/cosine/")
async def get_cosine_clusters():
    """
    Perform mathematical clustering and dynamic AI domain-agnostic labeling of the paper collection.
    """
    try:
        results = await execute_cluster_analysis()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing cluster analysis: {str(e)}")

@app.get("/analytics/clusters/hierarchical/")
async def get_hierarchical_clusters():
    """
    Perform Agglomerative Hierarchical Clustering and dynamic AI domain-agnostic labeling.
    """
    try:
        results = await execute_hierarchical_clustering()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing hierarchical cluster analysis: {str(e)}")

@app.get("/analytics/timeline/")
async def get_timeline():
    """
    Generate a chronological timeline of all completed research papers and a synthesized AI domain overview.
    """
    try:
        results = await execute_timeline_generation()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating research timeline: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)