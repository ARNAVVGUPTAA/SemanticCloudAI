import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DATA_PATH = os.getenv("CHROMA_DATA_PATH", "/app/chroma_data")

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Retrieve or create collections
directories_collection = chroma_client.get_or_create_collection(
    name="directories",
    metadata={"hnsw:space": "cosine"}
)

documents_collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Initialize LangChain Embeddings
# It uses the exact same model we were using before, but through LangChain
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
