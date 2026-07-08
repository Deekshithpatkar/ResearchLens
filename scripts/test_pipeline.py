#!/usr/bin/env python3
"""
Test script to verify the ResearchLens pipeline works end-to-end.
Tests: PDF extraction -> text chunking -> embedding generation
"""

import sys
import os
from pathlib import Path

# Add workspace root to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from backend.pdf_utils import extract_text_from_pdf
from backend.text_chunker import chunk_text
from backend.embedding_utils_simple import generate_embeddings


def test_pipeline():
    """Test the complete PDF -> embeddings pipeline"""
    
    # Use first sample PDF
    pdf_path = Path(__file__).parent.parent / "data" / "Attention is all you need.pdf"
    
    print(f"Testing pipeline with: {pdf_path.name}")
    print("=" * 70)
    
    # Step 1: Extract text
    print("\n1. Extracting text from PDF...")
    text = extract_text_from_pdf(str(pdf_path))
    print(f"   * Extracted {len(text)} characters")
    print(f"   Preview: {text[:200]}...")
    
    # Step 2: Chunk text
    print("\n2. Chunking text into segments...")
    chunks = chunk_text(text)
    print(f"   * Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:2]):
        print(f"   Chunk {i+1}: {len(chunk)} words")
    
    # Step 3: Generate embeddings
    print("\n3. Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    print(f"   * Generated {len(embeddings)} embeddings")
    print(f"   Embedding shape: {embeddings.shape} (items, dimensions)")
    
    # Verification
    print("\n" + "=" * 70)
    print("PIPELINE TEST SUCCESSFUL!")
    print(f"   - PDF text: {len(text)} chars")
    print(f"   - Chunks: {len(chunks)}")
    print(f"   - Embeddings: {embeddings.shape}")
    print("\nReady for Phase 1 implementation!")

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
