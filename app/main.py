import os
import shutil
import json
import logging
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import engine, get_db, Base
from .models import Document
from .tasks import process_document
from langchain_chroma import Chroma
from .vector_store import get_embeddings_model, chroma_client

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Pi Document Cloud")

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# Global LangChain embeddings model
embeddings_model = None

def get_query_embeddings():
    global embeddings_model
    if embeddings_model is None:
        logger.info("Loading LangChain Embeddings Model...")
        embeddings_model = get_embeddings_model()
    return embeddings_model

@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    extra_tags: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        user_dir = os.path.join(UPLOAD_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        file_location = os.path.join(user_dir, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        db_doc = Document(
            filename=file.filename,
            file_path=file_location,
            user_id=user_id,
            status="PROCESSING"
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)

        process_document.delay(db_doc.id, extra_tags)

        return {"id": db_doc.id, "filename": file.filename, "status": "PROCESSING"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_documents(
    query_text: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Semantic Search using SentenceTransformers.
    """
    try:
        # 1. Setup LangChain Vector Store
        embeddings = get_query_embeddings()
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="documents",
            embedding_function=embeddings
        )
        
        # 2. Extract keywords for context (optional display)
        context_tags = query_text.lower().split()
        
        # 3. LangChain Prompt-based Retrieval
        # Retrieve top 20 most similar documents using semantic similarity
        logger.info(f"LangChain Retrieval for Query: '{query_text}'")
        docs = vectorstore.similarity_search_with_relevance_scores(query_text, k=20)
        
        matches = []
        for doc, score in docs:
            matches.append({
                "id": doc.metadata.get("id"),
                "filename": doc.metadata.get("filename"),
                "tags": doc.metadata.get("tags", "").split(",") if doc.metadata.get("tags") else [],
                "category": doc.metadata.get("category"),
                "score": round(score, 4)
            })
            
        return {
            "interpreted_query": {"query": query_text, "context_tags": context_tags},
            "matches": matches
        }

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "tags": d.tags,
            "category": d.category,
            "upload_time": d.upload_time,
            "content_preview": d.content_text[:200] if d.content_text else None
        } for d in docs
    ]

@app.delete("/reset")
def reset_system(db: Session = Depends(get_db)):
    try:
        db.query(Document).delete()
        db.commit()
        
        # Reset ChromaDB collections
        try:
            chroma_client.delete_collection("documents")
            chroma_client.delete_collection("directories")
            chroma_client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})
            chroma_client.get_or_create_collection("directories", metadata={"hnsw:space": "cosine"})
        except Exception as ce:
            logger.warning(f"Error resetting chroma collections: {ce}")

        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e: pass
        return {"status": "Reset complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
