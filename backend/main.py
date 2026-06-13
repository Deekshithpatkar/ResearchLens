from fastapi import FastAPI, UploadFile, File
import shutil
from backend.pdf_utils import extract_text_from_pdf
from backend.text_chunker import chunk_text
from backend.embedding_utils import generate_embeddings

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ResearchLens Running"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = extract_text_from_pdf(file_path)

    chunks = chunk_text(extracted_text)
    embeddings = generate_embeddings(chunks)
    
    return {
    "filename": file.filename,
    "total_chunks": len(chunks),
    "embedding_dimension": len(embeddings[0]),
    "first_chunk_preview": chunks[0][:300]
}