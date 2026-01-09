import os
import shutil
import json
import logging
import requests
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from .database import engine, get_db, Base
from .models import Document
from .tasks import process_document

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Pi Document Cloud")

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
logger = logging.getLogger(__name__)

@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
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

        # Trigger Celery Task
        process_document.delay(db_doc.id)

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
    Natural language query for documents.
    1. Ask Ollama to extract keywords/filters.
    2. Search DB.
    """
    prompt = f"""
    Extract search keywords and category from this user query: "{query_text}".
    Return ONLY a JSON object with keys:
    - "keywords": list of strings (for text search)
    - "category": string or null (if user mentioned a specific category like 'invoice', 'receipt', 'contract')
    
    JSON:
    """
    
    try:
        # Call Ollama to interpret query
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=10
        )
        if response.status_code == 200:
            res_json = response.json()
            analysis = json.loads(res_json.get("response", "{}"))
            keywords = analysis.get("keywords", [])
            category = analysis.get("category")
        else:
            # Fallback to simple split
            keywords = query_text.split()
            category = None
            
    except Exception as e:
        logger.error(f"Ollama query interpretation failed: {e}")
        keywords = query_text.split()
        category = None

    # Construct DB Query
    db_query = db.query(Document)
    
    if category:
        db_query = db_query.filter(Document.category.ilike(f"%{category}%"))
    
    if keywords:
        # Simple naive search: check if any keyword is in content_text or filename
        # Postgres Full Text Search would be better, but keeping it simple for now.
        conditions = []
        for kw in keywords:
            conditions.append(Document.content_text.ilike(f"%{kw}%"))
            conditions.append(Document.filename.ilike(f"%{kw}%"))
            conditions.append(Document.summary.ilike(f"%{kw}%"))
        
        db_query = db_query.filter(or_(*conditions))

    results = db_query.all()
    
    return {
        "interpreted_query": {"keywords": keywords, "category": category},
        "matches": [
            {
                "id": d.id,
                "filename": d.filename,
                "summary": d.summary,
                "tags": d.tags,
                "category": d.category
            } for d in results
        ]
    }
