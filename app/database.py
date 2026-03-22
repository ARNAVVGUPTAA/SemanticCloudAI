from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Fallback to sqlite if running locally without Docker
default_db = "sqlite:///./doc_cloud.db" if os.getenv("LOCAL_MODE") else "postgresql://postgres:postgres@localhost:5432/doc_cloud"
DATABASE_URL = os.getenv("DATABASE_URL", default_db)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
