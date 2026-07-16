import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.database import engine, Base, get_db
from backend.models import User as DBUser, Paper as DBPaper
from backend.auth import hash_password, verify_password, create_access_token, get_current_user
from backend.config import (
    DATA_DIR,
    PAPERS_DIR,
    EMBEDDINGS_DIR,
    CHUNKS_DIR,
    collection,
    get_user_papers_dir,
    get_user_embeddings_dir,
    get_user_chunks_dir,
    get_user_profiles_dir,
    get_user_timelines_dir,
    get_user_clusters_dir
)
from backend.metadata_utils import load_metadata, save_metadata
from backend.search_utils import build_summary_text, execute_semantic_search
from backend.llm_utils import generate_response
from backend.pdf_utils import extract_text_from_pdf
from backend.text_chunker import chunk_text
from backend.embedding_utils_simple import generate_embeddings
from backend.profile_utils import extract_paper_profile
from backend.analytics_utils import (
    execute_global_analytics,
    execute_cluster_analysis,
    execute_timeline_generation,
    execute_hierarchical_clustering
)

from backend.migration import run_legacy_migration

# Initialize database tables
Base.metadata.create_all(bind=engine)

# Run legacy migration
run_legacy_migration()

app = FastAPI(
    title="ResearchLens API",
    description="PDF processing, semantic search, and RAG query engine",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRegister(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": "ResearchLens API v0.2.0",
        "status": "running",
        "endpoints": [
            "POST /auth/register",
            "POST /auth/login",
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

@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(DBUser).filter(DBUser.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed = hash_password(user_in.password)
    user = DBUser(email=user_in.email, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User registered successfully", "user_id": user.id}

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload-pdf/")
async def upload_pdf(
    file: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and process multiple PDF files.
    Extracts text, generates chunks, and creates embeddings for each.
    """
    results = []
    metadata = load_metadata(user_id=current_user.id)
    
    # Get user scoped paths
    user_papers_dir = get_user_papers_dir(current_user.id)
    user_chunks_dir = get_user_chunks_dir(current_user.id)
    user_embeddings_dir = get_user_embeddings_dir(current_user.id)
    
    for f in file:
        try:
            # Save uploaded file in user-scoped pdfs directory
            file_path = user_papers_dir / f.filename
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
            
            # Save chunks and embeddings locally in user-scoped directories
            chunks_file = user_chunks_dir / f"{paper_id}.json"
            with open(chunks_file, 'w', encoding="utf-8") as chunks_f:
                json.dump(chunks, chunks_f)
            
            embeddings_file = user_embeddings_dir / f"{paper_id}.npy"
            np.save(embeddings_file, embeddings)

            # Replace any older index entries for this user and paper before upserting fresh content
            collection.delete(where={"$and": [{"paper_id": paper_id}, {"user_id": current_user.id}]})

            # Save summary and chunk texts to ChromaDB with user_id metadata injection
            ids = [f"{paper_id}-summary"] + [f"{paper_id}-{idx}" for idx in range(len(chunks))]
            metadatas = [
                {
                    "user_id": current_user.id,
                    "paper_id": paper_id,
                    "chunk_type": "summary",
                    "chunk_index": -1,
                    "filename": f.filename,
                }
            ] + [
                {
                    "user_id": current_user.id,
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
            
            # Update/insert PostgreSQL paper metadata
            db_paper = db.query(DBPaper).filter(DBPaper.user_id == current_user.id, DBPaper.paper_id == paper_id).first()
            if not db_paper:
                db_paper = DBPaper(
                    user_id=current_user.id,
                    paper_id=paper_id,
                    filename=f.filename,
                    title=paper_id.replace("_", " "),
                    processing_status="completed",
                    chroma_collection="researchlens_gemini" if bool(os.environ.get("GEMINI_API_KEY")) else "researchlens"
                )
                db.add(db_paper)
            else:
                db_paper.filename = f.filename
                db_paper.upload_date = datetime.utcnow()
                db_paper.processing_status = "completed"
            db.commit()

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
            
            # Queue background task to extract structured paper profile, passing user_id
            if background_tasks:
                background_tasks.add_task(extract_paper_profile, paper_id, f.filename, False, current_user.id)
            
        except Exception as e:
            results.append({
                "filename": f.filename,
                "status": "error",
                "detail": str(e)
            })

    # Save the updated metadata once after the loop
    save_metadata(metadata, user_id=current_user.id)
    
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
def list_papers(current_user: DBUser = Depends(get_current_user)):
    """List all processed papers with metadata for the authenticated user"""
    metadata = load_metadata(user_id=current_user.id)
    return {
        "total_papers": len(metadata.get("papers", {})),
        "papers": metadata.get("papers", {})
    }

@app.delete("/papers/{paper_id}/")
def delete_paper(
    paper_id: str,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a paper and all associated local files and database indexes"""
    db_paper = db.query(DBPaper).filter(DBPaper.user_id == current_user.id, DBPaper.paper_id == paper_id).first()
    if not db_paper:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found or access denied")

    # Delete local files
    user_papers_dir = get_user_papers_dir(current_user.id)
    user_chunks_dir = get_user_chunks_dir(current_user.id)
    user_embeddings_dir = get_user_embeddings_dir(current_user.id)
    user_profiles_dir = get_user_profiles_dir(current_user.id)

    pdf_file = user_papers_dir / db_paper.filename
    chunks_file = user_chunks_dir / f"{paper_id}.json"
    embeddings_file = user_embeddings_dir / f"{paper_id}.npy"
    profile_file = user_profiles_dir / f"{paper_id}.json"

    for file_path in [pdf_file, chunks_file, embeddings_file, profile_file]:
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    # Delete from ChromaDB
    try:
        collection.delete(where={"$and": [{"paper_id": paper_id}, {"user_id": current_user.id}]})
    except Exception as e:
        print(f"Error deleting from ChromaDB: {e}")

    # Remove from user metadata JSON
    metadata = load_metadata(user_id=current_user.id)
    if "papers" in metadata and paper_id in metadata["papers"]:
        del metadata["papers"][paper_id]
        save_metadata(metadata, user_id=current_user.id)

    # Delete SQL record
    db.delete(db_paper)
    db.commit()

    return {"status": "success", "message": f"Paper {paper_id} deleted successfully"}

@app.post("/query/")
async def query_papers(
    query: str,
    paper_id: str = None,
    top_k: int = 8,
    current_user: DBUser = Depends(get_current_user)
):
    """
    Perform semantic search and generate a synthesized, polished response using Gemini.
    """
    try:
        search_results = await execute_semantic_search(
            query=query,
            paper_id=paper_id,
            top_k=top_k,
            user_id=current_user.id
        )
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
async def global_analytics(
    query: str,
    paper_ids: Optional[str] = Query(None),
    current_user: DBUser = Depends(get_current_user)
):
    """
    Perform global, multi-paper analytics comparison (Map-Reduce RAG) over paper profiles using query params.
    Example: ?query=compare objectives&paper_ids=all OR ?query=compare objectives&paper_ids=ALBERT,BERT
    """
    try:
        parsed_ids = None
        if paper_ids:
            parsed_ids = [pid.strip() for pid in paper_ids.split(",") if pid.strip()]
        result = await execute_global_analytics(query=query, paper_ids=parsed_ids, user_id=current_user.id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing analytics: {str(e)}")

@app.post("/analytics/reprocess/")
def reprocess_profiles(
    background_tasks: BackgroundTasks,
    paper_ids: Optional[str] = Query(None),
    current_user: DBUser = Depends(get_current_user)
):
    """
    Manually re-trigger structured paper profile extraction using query params.
    Example: ?paper_ids=ALBERT,BERT OR empty to reprocess all papers.
    """
    metadata = load_metadata(user_id=current_user.id)
    all_paper_ids = list(metadata.get("papers", {}).keys())

    target_ids = all_paper_ids
    if paper_ids:
        target_ids = [pid.strip() for pid in paper_ids.split(",") if pid.strip()]
        
    target_ids = [pid for pid in target_ids if pid in all_paper_ids]

    triggered = []
    for pid in target_ids:
        filename = metadata["papers"][pid].get("filename", f"{pid}.pdf")
        background_tasks.add_task(extract_paper_profile, pid, filename, False, current_user.id)
        triggered.append(pid)

    return {
        "status": "triggered",
        "message": f"Queued profile extraction background tasks for {len(triggered)} papers.",
        "papers": triggered
    }

@app.get("/analytics/clusters/cosine/")
async def get_cosine_clusters(current_user: DBUser = Depends(get_current_user)):
    """
    Perform mathematical clustering and dynamic AI domain-agnostic labeling of the paper collection.
    """
    try:
        results = await execute_cluster_analysis(user_id=current_user.id)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing cluster analysis: {str(e)}")

@app.get("/analytics/clusters/hierarchical/")
async def get_hierarchical_clusters(current_user: DBUser = Depends(get_current_user)):
    """
    Perform Agglomerative Hierarchical Clustering and dynamic AI domain-agnostic labeling.
    """
    try:
        results = await execute_hierarchical_clustering(user_id=current_user.id)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing hierarchical cluster analysis: {str(e)}")

@app.get("/analytics/timeline/")
async def get_timeline(current_user: DBUser = Depends(get_current_user)):
    """
    Generate a chronological timeline of all completed research papers and a synthesized AI domain overview.
    """
    try:
        results = await execute_timeline_generation(user_id=current_user.id)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating research timeline: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)