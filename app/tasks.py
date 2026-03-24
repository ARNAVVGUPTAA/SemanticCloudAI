import os
import logging
import json
import numpy as np
import torch
from PIL import Image
import pytesseract
from pypdf import PdfReader
from celery import Celery
from celery.signals import worker_process_init
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import Document
from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from .vector_store import directories_collection, documents_collection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("worker", broker=CELERY_BROKER_URL)

if os.getenv("LOCAL_MODE"):
    celery_app.conf.task_always_eager = True
    logger.info("LOCAL_MODE enabled: Celery tasks will run eagerly (synchronously) without a broker.")

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

@worker_process_init.connect
def init_worker(**kwargs):
    logger.info("Initializing worker process: pre-loading models")
    ModelManager.get_instance()

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
        all_text_chunks = []
        
        # Stream and Process
        # We extract standard named entities. Removed "number", "date", "topic" to reduce noise.
        labels_to_extract = ["person", "organization", "location"]
        
        # Buffer for SLM (First Page / 1000 chars)
        slm_context_buffer = ""
        
        # Initialize LangChain text splitter to count actual tokens.
        text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            chunk_overlap=40,
            tokens_per_chunk=384
        )
        
        for page_text in extract_text_stream(doc.file_path):
            if not page_text.strip():
                continue
                
            # Keep a bit of text for DB content_text (preview)
            if len(full_text_buffer) < 5: 
                full_text_buffer.append(page_text)
                
            # Accumulate text for SLM (approx first 2-3k chars is enough for context)
            if len(slm_context_buffer) < 2000:
                slm_context_buffer += " " + page_text
            
            # 1. Break down the massive page text into smaller individual chunks
            chunks = text_splitter.split_text(page_text)
            
            for text_chunk in chunks:
                if text_chunk.strip():
                    all_text_chunks.append(text_chunk)
        
        if not all_text_chunks:
            logger.warning("No text content could be processed.")
            doc.status = "FAILED"
            doc.content_text = "No extractable text found."
            db.commit()
            return

        # 2. Batch Encode Embeddings
        logger.info(f"Batch Encoding {len(all_text_chunks)} chunks for document {doc_id}")
        with torch.no_grad():
            doc_embedding_matrix = models.embed_model.encode(all_text_chunks, batch_size=16, show_progress_bar=False)
        chunk_embeddings = list(doc_embedding_matrix)

        # 3. Batch Extract Entities (Mini-batches for CPU optimization)
        batch_size = 16
        with torch.no_grad():
            if hasattr(models.ner_model, 'batch_predict_entities'):
                try:
                    batch_results = models.ner_model.batch_predict_entities(all_text_chunks, labels_to_extract, threshold=0.3)
                    for entities in batch_results:
                        for ent in entities:
                            all_tags.add(ent['text'].lower())
                except Exception as e:
                    logger.warning(f"Batch NER extraction failed: {e}")
            else:
                for i in range(0, len(all_text_chunks), batch_size):
                    mini_batch = all_text_chunks[i:i + batch_size]
                    try:
                        for text_chunk in mini_batch:
                            entities = models.ner_model.predict_entities(text_chunk, labels_to_extract, threshold=0.3)
                            for ent in entities:
                                all_tags.add(ent['text'].lower())
                    except Exception as e:
                        logger.warning(f"Mini-batch NER extraction failed: {e}")

        # Aggregate Results
        
        # 1. Document Embedding (Mean Pooling of Chunks)
        doc_embedding_matrix = np.vstack(chunk_embeddings)
        doc_embedding = np.mean(doc_embedding_matrix, axis=0)
        
        # 2. Semantic Understanding with SLM
        # We use the SLM on the beginning of the document to get the Type and Keywords
        
        best_category = "Document"
        slm_tags = []
        
        ''' TEMPORARY BYPASS: SLM Inference is disabled for stability/speed.
        if slm_context_buffer:
            try:
                # A. Identify Document Type
                prompt_type = f"Identify the specific document type (e.g. Statement of Purpose, Invoice, Research Paper, Resume) for this text: '{slm_context_buffer}'"
                input_ids = models.tokenizer(prompt_type, return_tensors="pt", truncation=True, max_length=512).input_ids.to(models.device)
                outputs = models.slm.generate(input_ids, max_length=50)
                doc_type_pred = models.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                if doc_type_pred and len(doc_type_pred) > 2:
                     best_category = doc_type_pred
                     all_tags.add(best_category)
                else:
                    # Fallback to zero-shot format detection if SLM fails/is vague
                    doc_emb_tensor = torch.tensor(doc_embedding, dtype=torch.float32).to(models.device)
                    format_scores = util.cos_sim(doc_emb_tensor, models.format_embeddings)[0]
                    best_format_idx = torch.argmax(format_scores).item()
                    best_category = models.formats[best_format_idx]

                # B. Generate Semantic Keywords
                prompt_tags = f"Generate 5 specific, comma-separated keywords or topics that describe this text: '{slm_context_buffer}'"
                input_ids = models.tokenizer(prompt_tags, return_tensors="pt", truncation=True, max_length=512).input_ids.to(models.device)
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
        '''
        
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
        
        # 4. Semantic Directory Logic (Thresholds: >85% merge, <40% branch)
        doc_emb_list = doc_embedding.tolist()
        assigned_category = best_category # Default from SLM
        
        # Query directories
        dir_results = directories_collection.query(
            query_embeddings=[doc_emb_list],
            n_results=1
        )
        
        if dir_results and dir_results["distances"] and len(dir_results["distances"][0]) > 0:
            # ChromaDB cosine distance: distance = 1.0 - cosine_similarity
            # Therefore similarity = 1.0 - distance
            distance = dir_results["distances"][0][0]
            similarity = 1.0 - distance
            closest_dir_name = dir_results["metadatas"][0][0]["name"]
            
            logger.info(f"Closest directory: {closest_dir_name} with similarity {similarity:.2f}")
            
            if similarity >= 0.85:
                # Merge: Perfect match
                assigned_category = closest_dir_name
                logger.info(f"Merge: Document mapped to existing directory '{assigned_category}'")
            elif similarity <= 0.40:
                # Branch: New concept
                assigned_category = best_category
                logger.info(f"Branch: Creating new directory '{assigned_category}' based on concept")
                # Add new directory to Chroma
                directories_collection.add(
                    ids=[assigned_category],
                    embeddings=[doc_emb_list],
                    metadatas=[{"name": assigned_category}]
                )
            else:
                # Middle-ground: Assign to closest, or we could strict branch. 
                # Request was "85% merge, 40% branch". We will assign to closest if it's kinda related.
                assigned_category = closest_dir_name
                logger.info(f"Assigning to moderately related directory '{assigned_category}'")
        else:
            # No directories exist yet, create the first one
            logger.info(f"First directory created: '{assigned_category}'")
            directories_collection.add(
                ids=[assigned_category],
                embeddings=[doc_emb_list],
                metadatas=[{"name": assigned_category}]
            )
            
        # Finalize Doc in DB
        doc.tags = list(set(final_tags)) # De-duplicate
        doc.category = assigned_category
        doc.content_text = "\n\n".join(full_text_buffer)[:5000] 
        doc.status = "COMPLETED"
        
        # Insert Document Vector into ChromaDB
        documents_collection.add(
            ids=[str(doc.id)],
            embeddings=[doc_emb_list],
            metadatas=[{
                "id": doc.id,
                "filename": doc.filename,
                "category": assigned_category,
                "tags": ",".join(doc.tags)
            }]
        )
        
        db.commit()
        logger.info(f"Finished processing document {doc_id}. Assigned: {assigned_category}")

    except Exception as e:
        logger.error(f"Critical error in task: {e}", exc_info=True)
        doc.status = "FAILED"
        doc.content_text = f"Error processing document: {str(e)}"
        db.commit()
    finally:
        db.close()
