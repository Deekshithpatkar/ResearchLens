#!/usr/bin/env python3
"""
Test script to verify LLM integration and the /query/ RAG endpoint.
"""

import sys
import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Add workspace root to path to allow absolute imports
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from backend.llm_utils import generate_response

BASE_URL = "http://127.0.0.1:8000"

def test_direct_llm():
    """Test llm_utils.generate_response directly with mock chunks"""
    print("\n[Step 1] Testing direct LLM response generation...")
    mock_chunks = [
        {
            "paper_id": "Deepfake_Detection_Paper",
            "chunk_type": "summary",
            "chunk_preview": "This paper presents a new Hindi audio-video deepfake dataset (HAV-DF) and evaluates standard architectures like CNN-LSTM for deepfake detection, achieving 87% accuracy."
        },
        {
            "paper_id": "Deepfake_Detection_Paper",
            "chunk_type": "chunk",
            "chunk_preview": "We collected 500 real and 500 fake Hindi video clips. The audio was synthesized using popular text-to-speech tools, and the video was manipulated using deepfake generation tools like First Order Motion Model."
        }
    ]
    query = "What dataset is introduced in the paper and what is the accuracy?"
    print(f"   Query: '{query}'")
    answer = generate_response(query, mock_chunks)
    print("\n--- LLM Response ---")
    print(answer)
    print("--------------------")
    
    if "Error:" in answer or "not set" in answer:
        print("   * LLM execution was skipped or failed (likely missing API key).")
        return False
    else:
        print("   * Direct LLM generation succeeded!")
        return True

def test_query_endpoint():
    """Test the /query/ API endpoint"""
    print("\n[Step 2] Testing /query/ RAG endpoint...")
    query = "Explain deepfake detection methods used in HAV-DF"
    try:
        response = requests.post(
            f"{BASE_URL}/query/",
            params={"query": query, "top_k": 3}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("\n--- API Query Answer ---")
            print(data.get("answer"))
            print("------------------------")
            print(f"   * Chunks retrieved: {len(data.get('chunks', []))}")
            return True
        else:
            print(f"   x Error Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("   * FastAPI server is not running at http://127.0.0.1:8000. Skipping endpoint test.")
        print("   (To test the endpoint, run 'uvicorn backend.main:app --reload' in another terminal)")
        return None

def main():
    print("=" * 70)
    print("ResearchLens LLM Integration Test Suite")
    print("=" * 70)
    
    # Load dotenv
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("x WARNING: GEMINI_API_KEY environment variable is NOT set in .env")
        print("   Please create a .env file and set GEMINI_API_KEY to test the LLM functions.")
    else:
        print("* GEMINI_API_KEY detected in environment.")
        
    direct_ok = test_direct_llm()
    endpoint_ok = test_query_endpoint()
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Direct LLM Test: {'PASSED' if direct_ok else 'FAILED/SKIPPED'}")
    if endpoint_ok is True:
        print("RAG Endpoint Test: PASSED")
    elif endpoint_ok is False:
        print("RAG Endpoint Test: FAILED")
    else:
        print("RAG Endpoint Test: SKIPPED (Server not running)")

if __name__ == "__main__":
    main()
