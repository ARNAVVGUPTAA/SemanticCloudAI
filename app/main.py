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
    extra_tags: Optional[str] = Form(None),
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
    Natural language query for documents.
    1. Ask Ollama to generate 'context-specific tags' from the query.
    2. Ask Ollama if the user is asking for 'latest' or 'recent' documents.
    3. Search DB by matching generated tags against document tags.
    """
    prompt = f"""
    Analyze this user query: "{query_text}".
    Return ONLY a JSON object with keys:
    - "context_tags": list of strings (tags that would likely be on the documents the user is looking for).
    - "sort_by_recency": boolean (true if user mentions 'latest', 'recent', 'newest', 'last', etc.)
    - "category": string or null (if user mentioned a specific category)
    
    JSON:
    """
    
    try:
        # Call Ollama to interpret query
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=30
        )
        if response.status_code == 200:
            res_json = response.json()
            analysis = json.loads(res_json.get("response", "{}"))
            context_tags = analysis.get("context_tags", [])
            sort_by_recency = analysis.get("sort_by_recency", False)
            category = analysis.get("category")
        else:
            # Fallback
            context_tags = query_text.split()
            sort_by_recency = False
            category = None
            
    except Exception as e:
        logger.error(f"Ollama query interpretation failed: {e}")
        context_tags = query_text.split()
        sort_by_recency = False
        category = None

    # Construct DB Query
    db_query = db.query(Document)
    
    if category:
        db_query = db_query.filter(Document.category.ilike(f"%{category}%"))
    
    # Filter by Tags (Naively check if document tags contain ANY of the context tags)
    # Since tags is JSON, we might need a more complex query in real production (e.g. Postgres @> operator),
    # but for SQLite/Simple implementation, we might have to fetch and filter or use basic string matching if stored as string.
    # However, model defines tags as JSON.
    # For simplicity and compatibility with the current setup (likely SQLite or basic Postgres usage without specific JSON types setup in code):
    # We will fetch all and filter in Python if the dataset is small, OR use a LIKE query if we cast to text.
    # Let's try a robust approach: filter in Python for now as dataset is likely small for this PoC.
    # ideally: db_query = db_query.filter(func.json_each(Document.tags).has(func.any(context_tags))) - depends on DB
    
    # Let's start with all docs passed previous filters
    candidates = db_query.all()
    
    matches = []
    for doc in candidates:
        score = 0
        doc_tags_set = set(doc.tags) if doc.tags else set()
        
        # Check tag overlap
        # Normalized comparison
        doc_tags_normalized = {str(t).lower() for t in doc_tags_set}
        
        for q_tag in context_tags:
            if q_tag.lower() in doc_tags_normalized:
                score += 1
        
        # Also check filename/content as fallback
        if any(q_tag.lower() in doc.filename.lower() for q_tag in context_tags):
            score += 0.5

        if score > 0 or not context_tags:
            matches.append((doc, score))
            
    # Sort matches
    if sort_by_recency:
        # Sort by upload_time desc
        matches.sort(key=lambda x: x[0].upload_time or x[0].id, reverse=True)
    else:
        # Sort by score desc
        matches.sort(key=lambda x: x[1], reverse=True)
        
    results = [m[0] for m in matches]
    
    return {
        "interpreted_query": {"context_tags": context_tags, "sort_by_recency": sort_by_recency, "category": category},
        "matches": [
            {
                "id": d.id,
                "filename": d.filename,
                # "summary": d.summary, # Removed summary usage
                "tags": d.tags,
                "category": d.category,
                "upload_time": d.upload_time
            } for d in results
        ]
    }

@app.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    """
    Get all documents from the database.
    Useful for debugging and inspection.
    """
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
