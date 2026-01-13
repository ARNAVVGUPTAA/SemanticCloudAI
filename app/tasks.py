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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

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
        # Using all-mpnet-base-v2: ~420MB, 768 dims. Best quality for general tasks.
        self.embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=self.device)

        # 2. NER Model (GLiNER)
        # Using gliner_medium-v2.1: ~300MB. Better, deeper entity extraction.
        self.ner_model = GLiNER.from_pretrained('urchade/gliner_medium-v2.1').to(self.device)
        self.ner_model.eval()

        # 3. Small Language Model (LaMini-Flan-T5-248M)
        # ~500MB RAM. Provides semantic understanding/instruction following.
        logger.info("Loading SLM (LaMini-Flan-T5-248M)...")
        self.tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
        self.slm = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M").to(self.device)
        self.slm.eval()

        # 4. Taxonomy (Optional fallback)
        try:
            with open("app/taxonomy.json", "r") as f:
                data = json.load(f)
                self.formats = data.get("formats", ["Document"])
        except Exception as e:
            logger.warning(f"Could not load taxonomy.json, using defaults: {e}")
            self.formats = ["Document", "Invoice", "Receipt", "Paper", "Book"]

        # Pre-compute embeddings for formats (fallback axis)
        self.format_embeddings = self.embed_model.encode(self.formats, convert_to_tensor=True)
        
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
        # We extract standard named entities. Removed "number", "date", "topic" to reduce noise.
        labels_to_extract = ["person", "organization", "location"]
        
        # Buffer for SLM (First Page / 1000 chars)
        slm_context_buffer = ""
        
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
                
            # Accumulate text for SLM (approx first 2-3k chars is enough for context)
            if len(slm_context_buffer) < 2000:
                slm_context_buffer += " " + text_chunk
        
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
        
        # 2. Semantic Understanding with SLM
        # We use the SLM on the beginning of the document to get the Type and Keywords
        
        best_category = "Document"
        slm_tags = []
        
        if slm_context_buffer:
            try:
                # Truncate to avoid max length issues (512 tokens approx 2000 chars)
                input_text = slm_context_buffer[:2000]
                
                # A. Identify Document Type
                prompt_type = f"Identify the specific document type (e.g. Statement of Purpose, Invoice, Research Paper, Resume) for this text: '{input_text}'"
                input_ids = models.tokenizer(prompt_type, return_tensors="pt").input_ids.to(models.device)
                outputs = models.slm.generate(input_ids, max_length=50)
                doc_type_pred = models.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                if doc_type_pred and len(doc_type_pred) > 2:
                     best_category = doc_type_pred
                     all_tags.add(best_category)
                else:
                    # Fallback to zero-shot format detection if SLM fails/is vague
                    format_scores = util.cos_sim(doc_emb_tensor, models.format_embeddings)[0]
                    best_format_idx = torch.argmax(format_scores).item()
                    best_category = models.formats[best_format_idx]

                # B. Generate Semantic Keywords
                prompt_tags = f"Generate 5 specific, comma-separated keywords or topics that describe this text: '{input_text}'"
                input_ids = models.tokenizer(prompt_tags, return_tensors="pt").input_ids.to(models.device)
                outputs = models.slm.generate(input_ids, max_length=100)
                keywords_text = models.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Parse keywords
                for kw in keywords_text.split(','):
                    clean_kw = kw.strip()
                    if clean_kw:
                        slm_tags.append(clean_kw)
                        all_tags.add(clean_kw)
                        
            except Exception as e:
                logger.warning(f"SLM inference failed: {e}")
                # Fallback to default
        
        # 3. Final Tag Cleaning
        final_tags = []
        for tag in all_tags:
            # Normalize
            t = tag.strip()
            # Remove junk: 
            # - Short tags (<3 chars)
            # - Purely numeric (e.g "2024", "1")
            # - Special chars only
            if len(t) < 3: continue
            if t.isdigit(): continue
            if re.match(r'^[0-9\W]+$', t): continue # Only numbers and symbols
            
            final_tags.append(t)
        
        # Finalize Doc
        # Finalize Doc
        doc.tags = list(set(final_tags)) # De-duplicate
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
