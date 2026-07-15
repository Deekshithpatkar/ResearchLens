import pytest
import io
import os
import shutil
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.main import app
from backend.database import Base, get_db
from backend.models import User, Paper
from backend.config import get_user_data_dir

# Use a file-based temporary test database to keep tables persistent across connections
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_temp.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="module")
def client():
    # Remove any existing temp test DB
    if os.path.exists("./test_temp.db"):
        try:
            os.unlink("./test_temp.db")
        except Exception:
            pass
            
    Base.metadata.create_all(bind=engine)
    
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
            
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()
    
    # Delete temp test DB file
    if os.path.exists("./test_temp.db"):
        try:
            os.unlink("./test_temp.db")
        except Exception:
            pass

from unittest.mock import patch, AsyncMock

def test_user_flow_and_isolation(client):
    mock_embeddings = lambda docs, for_query=False: np.zeros((len(docs), 3072))
    
    with patch("backend.main.extract_text_from_pdf", return_value="Abstract: This is a paper about something. Introduction: content."), \
         patch("backend.main.generate_embeddings", side_effect=mock_embeddings), \
         patch("backend.main.generate_response", return_value="Mocked response text"), \
         patch("backend.main.extract_paper_profile") as mock_extract:
         
        # 1. Register User A and User B
        res = client.post("/auth/register", json={"email": "usera@example.com", "password": "passwordA123!"})
        assert res.status_code == 201
        user_a_id = res.json()["user_id"]
    
        res = client.post("/auth/register", json={"email": "userb@example.com", "password": "passwordB123!"})
        assert res.status_code == 201
        user_b_id = res.json()["user_id"]
    
        # 2. Login User A
        res = client.post("/auth/login", data={"username": "usera@example.com", "password": "passwordA123!"})
        assert res.status_code == 200
        token_a = res.json()["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}
    
        # 3. Login User B
        res = client.post("/auth/login", data={"username": "userb@example.com", "password": "passwordB123!"})
        assert res.status_code == 200
        token_b = res.json()["access_token"]
        headers_b = {"Authorization": f"Bearer {token_b}"}
    
        # 4. Upload Paper A for User A
        pdf_content = b"%PDF-1.4 Mock PDF Content with introduction and abstract."
        file_a = io.BytesIO(pdf_content)
        res = client.post("/upload-pdf/", files={"file": ("Paper_A.pdf", file_a, "application/pdf")}, headers=headers_a)
        if res.status_code != 200:
            print("USER A UPLOAD FAIL:", res.status_code, res.text)
        assert res.status_code == 200
        assert res.json()["status"] == "completed"
    
        # 5. Upload Paper B for User B
        file_b = io.BytesIO(pdf_content)
        res = client.post("/upload-pdf/", files={"file": ("Paper_B.pdf", file_b, "application/pdf")}, headers=headers_b)
        if res.status_code != 200:
            print("USER B UPLOAD FAIL:", res.status_code, res.text)
        assert res.status_code == 200
        assert res.json()["status"] == "completed"
    
        # 6. Verify User A lists only Paper A
        res = client.get("/papers/", headers=headers_a)
        assert res.status_code == 200
        papers_a = res.json()["papers"]
        assert "Paper_A" in papers_a
        assert "Paper_B" not in papers_a
    
        # 7. Verify User B lists only Paper B
        res = client.get("/papers/", headers=headers_b)
        assert res.status_code == 200
        papers_b = res.json()["papers"]
        assert "Paper_B" in papers_b
        assert "Paper_A" not in papers_b
    
        # 8. Verify User A cannot query Paper B
        # Query for User A searching for Paper B
        # In endpoint query_papers, Query is passed as form data or json or query param. Let's see:
        # FastAPI @app.post("/query/") async def query_papers(query: str, paper_id: str = None, top_k: int = 8)
        # It takes query parameters! So let's send query parameters.
        res = client.post("/query/?query=test&paper_id=Paper_B", headers=headers_a)
        assert res.status_code == 404
    
        # 9. Verify User A cannot delete Paper B
        res = client.delete("/papers/Paper_B/", headers=headers_a)
        assert res.status_code == 404
    
        # 10. Verify User A can delete Paper A successfully
        res = client.delete("/papers/Paper_A/", headers=headers_a)
        assert res.status_code == 200
    
        # Clean up directories generated for testing
        for uid in [user_a_id, user_b_id]:
            user_dir = get_user_data_dir(uid)
            if user_dir.exists():
                shutil.rmtree(user_dir)
