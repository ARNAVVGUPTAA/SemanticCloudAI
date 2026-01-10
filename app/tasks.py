import os
import io
import json
import logging
import requests
from PIL import Image
import pytesseract
from pypdf import PdfReader
from celery import Celery
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

celery_app = Celery("worker", broker=CELERY_BROKER_URL)

def get_db_session():
    return SessionLocal()

@celery_app.task
def process_document(doc_id: int, extra_tags: str = None):
    """
    Background task to process uploaded documents.
    1. Extract text (OCR/PDF).
    2. Analyze with Ollama.
    3. Update Database.
    """
    db = get_db_session()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    
    if not doc:
        logger.error(f"Document {doc_id} not found.")
        db.close()
        return

    try:
        logger.info(f"Processing document {doc_id}: {doc.filename}")
        file_path = doc.file_path
        extracted_text = ""

        # 1. Text Extraction
        if file_path.lower().endswith('.pdf'):
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    extracted_text += page.extract_text() + "\n"
            except Exception as e:
                logger.error(f"Error reading PDF: {e}")
        else:
            # Assume image
            try:
                image = Image.open(file_path)
                extracted_text = pytesseract.image_to_string(image)
            except Exception as e:
                logger.error(f"Error reading Image: {e}")

        if not extracted_text.strip():
            logger.warning("No text extracted.")
            doc.status = "FAILED"
            doc.content_text = "No text could be extracted."
            db.commit()
            return

        doc.content_text = extracted_text
        
        # 2. Ollama Analysis & Embedding
        
        # Helper to get embeddings
        def get_embedding(text):
            try:
                # Try nomic-embed-text first, then llama3.2:1b
                model = "nomic-embed-text"
                req_body = {"model": model, "prompt": text}
                res = requests.post(f"{OLLAMA_HOST}/api/embeddings", json=req_body, timeout=60)
                if res.status_code != 200:
                    model = "llama3.2:1b"
                    req_body["model"] = model
                    res = requests.post(f"{OLLAMA_HOST}/api/embeddings", json=req_body, timeout=60)
                
                if res.status_code == 200:
                    return res.json().get("embedding")
                return None
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                return None

        prompt = f"""
        Analyze the following text from a document.
        
        prompt = f"""
        Analyze the following text from a document.
        
        Goal: Extract exhaustive metadata for a search engine. Quantity and precision matter.
        
        Return ONLY a JSON object with the following keys:
        - "tags": A list of strings.
          CRITICAL INSTRUCTIONS FOR TAGS:
          1. Extract AT LEAST 20 tags if the content allows.
          2. Extract EVERY proper noun (Person names, Company names, Locations, Product names).
          3. Extract ALL dates (years, specific dates like "2023", "January").
          4. Extract ALL numeric identifiers (Invoice numbers, Account numbers).
          5. Extract Document Type (e.g. "Invoice", "Receipt", "Bank Statement", "Notes").
          6. Extract inferred topics (e.g. "Networking", "Finance", "Groceries").
          7. Be granular. Don't summarize, extract.
        - "category": A single broad category for this document (e.g., "Finance", "Education", "Legal", "Personal", "Work").
          refrain from making new categories or types that mean very similar to one of the previous ones
        
        Do not include any other text, markdown formatting, or explanations. just the JSON.
        
        TEXT:
        {extracted_text[:6000]} 
        """
        # Truncate text to avoid context window issues if too large, 
        # though llama3.2 has decent context. 4000 chars is safe start.

        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json" # Force JSON mode if supported by model/version
                },
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            analysis_text = result.get("response", "{}")
            
            # Parse JSON
            try:
                metadata = json.loads(analysis_text)
                
                tags = metadata.get("tags", [])
                
                # Add extra tags if provided
                if extra_tags:
                    tags.extend([t.strip() for t in extra_tags.split(",") if t.strip()])
                
                # Add timestamp tag
                if doc.upload_time:
                    # Format as timestamp:YYYY-MM-DD
                    ts_tag = f"timestamp:{doc.upload_time.strftime('%Y-%m-%d')}"
                    tags.append(ts_tag)

                doc.tags = tags
                doc.category = metadata.get("category")
                
                # Generate Embedding using combined text representation
                # content + tags + category gives a rich semantic representation
                embedding_text = f"{doc.filename} {doc.category} {' '.join(tags)} {extracted_text[:1000]}"
                doc.embedding = get_embedding(embedding_text)
                
                doc.status = "COMPLETED"
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Ollama JSON response: {analysis_text}")
                doc.status = "COMPLETED" # Still completed, just missing metadata
                doc.tags = ["parsing_error"]

        except Exception as e:
            logger.error(f"Ollama inference failed: {e}")
            doc.status = "FAILED"
        
        db.commit()
        logger.info(f"Finished processing document {doc_id}")

    except Exception as e:
        logger.error(f"Critical error in task: {e}")
        doc.status = "FAILED"
        db.commit()
    finally:
        db.close()
