import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database import Base
from backend.main import app, get_db

# Use test temp database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_chats_temp.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def register_and_login(email, password):
    client.post("/auth/register", json={"email": email, "password": password})
    response = client.post("/auth/login", data={"username": email, "password": password})
    return response.json()["access_token"]

def test_chat_endpoints():
    token = register_and_login("chatuser@example.com", "Password123")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. List sessions (initially empty)
    response = client.get("/chats/", headers=headers)
    assert response.status_code == 200
    assert len(response.json()) == 0

    # 2. Create chat session
    response = client.post("/chats/", json={"title": "Test Chat", "chat_type": "rag"}, headers=headers)
    assert response.status_code == 200
    session = response.json()
    assert session["title"] == "Test Chat"
    assert session["chat_type"] == "rag"
    session_id = session["id"]

    # 3. List sessions (now containing 1)
    response = client.get("/chats/", headers=headers)
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["id"] == session_id

    # 4. Get messages (initially empty)
    response = client.get(f"/chats/{session_id}/messages/", headers=headers)
    assert response.status_code == 200
    assert len(response.json()) == 0

    # 5. Delete chat session
    response = client.delete(f"/chats/{session_id}/", headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # 6. Verify deleted
    response = client.get("/chats/", headers=headers)
    assert len(response.json()) == 0
