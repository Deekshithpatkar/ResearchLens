import json
from datetime import datetime
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import DATA_DIR, PAPERS_DIR, EMBEDDINGS_DIR, CHUNKS_DIR, collection
from backend.metadata_utils import load_metadata, save_metadata
from backend.search_utils import build_summary_text, execute_semantic_search
from backend.llm_utils import generate_response
from backend.pdf_utils import extract_text_from_pdf
from backend.text_chunker import chunk_text
from backend.embedding_utils_simple import generate_embeddings

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
            "POST /search/",
            "POST /query/"
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
        file_path = PAPERS_DIR / file.filename
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

@app.post("/search/")
def semantic_search(query: str, paper_id: str = None, top_k: int = 5):
    """Perform semantic search over paper chunks."""
    try:
        search_results = execute_semantic_search(query=query, paper_id=paper_id, top_k=top_k)
        return search_results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.post("/query/")
def query_papers(query: str, paper_id: str = None, top_k: int = 8):
    """
    Perform semantic search and generate a synthesized, polished response using Gemini.
    """
    try:
        search_results = execute_semantic_search(query=query, paper_id=paper_id, top_k=top_k)
        chunks = search_results.get("results", [])
        answer = generate_response(query=query, chunks=chunks)
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)