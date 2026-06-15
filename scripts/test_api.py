#!/usr/bin/env python3
"""
Test script for ResearchLens API endpoints.
Run this while the FastAPI server is running.
"""

import requests
import json
from pathlib import Path
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n1️⃣ Testing Health Check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_upload_pdf():
    """Test PDF upload and processing"""
    print("\n2️⃣ Testing PDF Upload...")
    
    pdf_path = Path(__file__).parent.parent / "data" / "Hindi audio video Deepfake (HAV-DF) A Hindi language-based.pdf"
    
    if not pdf_path.exists():
        print(f"   ❌ PDF not found: {pdf_path}")
        return False
    
    with open(pdf_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/upload-pdf/", files=files)
    
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Paper ID: {data.get('paper_id')}")
        print(f"   ✓ Chunks: {data.get('total_chunks')}")
        print(f"   ✓ Embedding Dimension: {data.get('embedding_dimension')}")
        return True
    else:
        print(f"   ❌ Error: {response.text}")
        return False

def test_list_papers():
    """Test listing papers"""
    print("\n3️⃣ Testing List Papers...")
    response = requests.get(f"{BASE_URL}/papers/")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Total papers: {data.get('total_papers')}")
    if data.get('papers'):
        for paper_id, info in data['papers'].items():
            print(f"   • {paper_id}: {info.get('num_chunks')} chunks")
    return response.status_code == 200

def test_search(query: str):
    """Test semantic search"""
    print(f"\n4️⃣ Testing Semantic Search...")
    print(f"   Query: '{query}'")
    
    response = requests.post(
        f"{BASE_URL}/search/",
        params={"query": query, "top_k": 3}
    )
    
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Results found: {data.get('num_results')}")
        for i, result in enumerate(data.get('results', []), 1):
            print(f"   {i}. Similarity: {result['similarity_score']:.4f}")
            print(f"      {result['chunk_preview'][:100]}...")
        return True
    else:
        print(f"   ❌ Error: {response.text}")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("🔬 ResearchLens API Test Suite")
    print("=" * 70)
    
    tests = [
        ("Health Check", test_health_check),
        ("PDF Upload", test_upload_pdf),
        ("List Papers", test_list_papers),
        ("Semantic Search", lambda: test_search("deepfake detection"))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
            time.sleep(0.5)  # Small delay between requests
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! ResearchLens API is ready for Phase 1 MVP.")
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("   Make sure the FastAPI server is running on http://127.0.0.1:8000")
        print("   Run: uvicorn backend.main:app --reload --port 8000")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
