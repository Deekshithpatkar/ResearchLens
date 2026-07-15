import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# Explicitly load environment variables from .env
load_dotenv()

from sqlalchemy.engine import make_url

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./researchlens.db")

# Parse connection URL safely (handles encoded password characters like %40)
db_url = make_url(DATABASE_URL)

# For SQLite, allow multithreading access
connect_args = {}
if db_url.drivername.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(db_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
