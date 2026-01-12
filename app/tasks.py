import os
import logging
import json
import numpy as np
import torch
from PIL import Image
import pytesseract
from pypdf import PdfReader
from celery import Celery
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import Document
from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("worker", broker=CELERY_BROKER_URL)

# Model Manager to handle lazy loading within the worker process
class ModelManager:
    _instance = None

    def __init__(self):
        logger.info("Loading models... (This may take a moment on first run)")
        self.device = "cpu"  # Force CPU for memory constraints/compatibility

        # 1. Embedding Model (SentenceTransformers)
        # Using all-MiniLM-L6-v2: ~80MB, 384 dims. Fast, efficient.
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # 2. NER Model (GLiNER)
        # Using gliner_small-v2.1: ~150MB. Good zero-shot performance.
        self.ner_model = GLiNER.from_pretrained('gliner-community/gliner_small-v2.1').to(self.device)
        self.ner_model.eval()

        # 3. Taxonomy for Zero-Shot Classification
        self.taxonomy = [
            "Finance", "Legal", "Invoice", "Receipt", 
            "Technical", "Research", "Medical", 
            "Personal", "Creative", "Administrative"
        ]
        # Pre-compute taxonomy embeddings
        self.taxonomy_embeddings = self.embed_model.encode(self.taxonomy, convert_to_tensor=True)
        
        logger.info("Models loaded successfully.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_db_session():
    return SessionLocal()

def extract_text_stream(file_path):
    """
    Generator that yields chunks of text from the file.
    Keeps memory footprint low by not loading the whole file.
    """
    if file_path.lower().endswith('.pdf'):
        try:
            reader = PdfReader(file_path)
            # Yield page by page
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    yield text
        except Exception as e:
            logger.error(f"Error reading PDF stream: {e}")
    else:
        # Fallback for images (not streamable in same way, but usually smaller)
        try:
            image = Image.open(file_path)
            yield pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Error reading Image: {e}")

@celery_app.task
def process_document(doc_id: int, extra_tags: str = None):
    db = get_db_session()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    
    if not doc:
        logger.error(f"Document {doc_id} not found.")
        db.close()
        return

    try:
        logger.info(f"Processing document {doc_id}: {doc.filename}")
        
        # Load Models (Lazy)
        models = ModelManager.get_instance()
        
        all_tags = set()
        if extra_tags:
            all_tags.update([t.strip() for t in extra_tags.split(",") if t.strip()])
        
        if doc.upload_time:
            all_tags.add(f"year:{doc.upload_time.year}")

        chunk_embeddings = []
        full_text_buffer = [] 
        
        # Stream and Process
        # We extract standard named entities + 'topic' (abstract)
        labels_to_extract = ["person", "organization", "location", "date", "number", "topic"]
        
        for text_chunk in extract_text_stream(doc.file_path):
            if not text_chunk.strip():
                continue
            
            # 1. Embed Chunk (using SentenceTransformer)
            embedding = models.embed_model.encode(text_chunk)
            chunk_embeddings.append(embedding)
            
            # 2. Extract Entities (using GLiNER)
            # Threshold 0.3-0.4 is usually good for small model
            try:
                entities = models.ner_model.predict_entities(text_chunk, labels_to_extract, threshold=0.3)
                for ent in entities:
                    all_tags.add(ent['text'].lower())
            except Exception as e:
                logger.warning(f"NER extraction failed for chunk: {e}")
            
            # Keep a bit of text for DB content_text (preview)
            if len(full_text_buffer) < 5: 
                full_text_buffer.append(text_chunk)
        
        if not chunk_embeddings:
            logger.warning("No text content could be processed.")
            doc.status = "FAILED"
            doc.content_text = "No extractable text found."
            db.commit()
            return

        # Aggregate Results
        
        # 1. Document Embedding (Mean Pooling of Chunks)
        doc_embedding_matrix = np.vstack(chunk_embeddings)
        doc_embedding = np.mean(doc_embedding_matrix, axis=0)
        
        # 2. Zero-Shot Categorization
        doc_emb_tensor = torch.tensor(doc_embedding).unsqueeze(0).to(models.device)
        sim_scores = util.cos_sim(doc_emb_tensor, models.taxonomy_embeddings)[0]
        
        best_cat_idx = torch.argmax(sim_scores).item()
        best_category = models.taxonomy[best_cat_idx]
        
        # Finalize Doc
        doc.tags = list(all_tags)
        doc.category = best_category
        # Convert numpy/tensor to list for JSON storage
        doc.embedding = doc_embedding.tolist()
        doc.content_text = "\n\n".join(full_text_buffer)[:5000] 
        doc.status = "COMPLETED"
        
        db.commit()
        logger.info(f"Finished processing document {doc_id}. Category: {best_category}")

    except Exception as e:
        logger.error(f"Critical error in task: {e}", exc_info=True)
        doc.status = "FAILED"
        doc.content_text = f"Error processing document: {str(e)}"
        db.commit()
    finally:
        db.close()
