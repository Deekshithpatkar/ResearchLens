import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    papers = relationship("Paper", back_populates="user", cascade="all, delete-orphan")

class Paper(Base):
    __tablename__ = "papers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    title = Column(String(255), nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    processing_status = Column(String(50), default="processing", nullable=False)
    chroma_collection = Column(String(100), nullable=True)

    user = relationship("User", back_populates="papers")
