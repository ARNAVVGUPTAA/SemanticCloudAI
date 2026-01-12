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
from sentence_transformers import SentenceTransformer, util
import torch

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Pi Document Cloud")

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# Global Embedding Model for Queries (Lazy Load)
query_model = None

def get_query_model():
    """Single instance of embedding model for the API process"""
    global query_model
    if query_model is None:
        # Must match tasks.py model
        logger.info("Loading Query Embedding Model...")
        query_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cpu")
    return query_model

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
        model = get_query_model()
        
        # 1. Embed Query
        query_embedding = model.encode(query_text)
        
        # 2. Extract keywords for hybrid filtering (simple split for now)
        # We could also use GLiNER here if we wanted to extract specific entities
        context_tags = query_text.lower().split()
        
        # 3. Fetch Candidates
        candidates = db.query(Document).filter(Document.status == "COMPLETED").all()
        
        results = []
        if not candidates:
            return {"interpreted_query": {"context_tags": context_tags}, "matches": []}

        # Vector Search
        # Prepare doc vectors
        doc_ids = []
        doc_vecs = []
        valid_candidates = []
        
        for doc in candidates:
            if doc.embedding and len(doc.embedding) > 0:
                doc_vecs.append(doc.embedding)
                doc_ids.append(doc.id)
                valid_candidates.append(doc)
        
        if not doc_vecs:
             return {"interpreted_query": {"context_tags": context_tags}, "matches": []}

        # Compute Similarities (Vectorized)
        # query_embedding: (384,)
        # doc_vecs: (N, 384)
        query_vec = torch.tensor(query_embedding).unsqueeze(0) # (1, 384)
        doc_matrix = torch.tensor(doc_vecs) # (N, 384)
        
        # Cosine Similarity
        cosine_scores = util.cos_sim(query_vec, doc_matrix)[0] # (N,)
        
        # Sort and Format
        matches = []
        for idx, score in enumerate(cosine_scores):
            doc = valid_candidates[idx]
            final_score = score.item()
            
            # Simple keyword boost
            # If filename or tags contain query terms, boost slightly
            boost = 0
            if any(t in doc.filename.lower() for t in context_tags):
                boost += 0.1
            
            matches.append((doc, final_score + boost))
            
        # Top Results
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:20]
        
        return {
            "interpreted_query": {"context_tags": context_tags},
            "matches": [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "tags": d.tags,
                    "category": d.category,
                    "score": round(s, 4),
                    "upload_time": d.upload_time
                } for d, s in top_matches
            ]
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
