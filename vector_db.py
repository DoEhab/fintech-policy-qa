# vector_db.py
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

client = QdrantClient(path="./qdrant_data")

# Dynamically name the collection based on the embedding model!
USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
COLLECTION_NAME = "fintech_pci_local" if USE_LOCAL else "fintech_pci_cohere"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
MODEL_DIMENSIONS = {
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "all-MiniLM-L6-v2": 384,
    "embed-english-v3.0": 1024,       # Cohere
    "embed-multilingual-v3.0": 1024   # Cohere
}
DIMENSION = MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 768)

# ... (keep your setup_qdrant_collection and upload_to_qdrant functions exactly the same) ...
def setup_qdrant_collection():
    """Creates the collection if it doesn't exist."""
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        logging.info(f"Creating new Qdrant collection: {COLLECTION_NAME}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
        )
        logging.info("Collection created successfully!")
    else:
        logging.info(f"Collection {COLLECTION_NAME} already exists.")

def upload_to_qdrant(chunks, embeddings, source_filename):
    """Formats and uploads the chunks and embeddings to Qdrant."""
    setup_qdrant_collection()
    
    logging.info(f"Preparing {len(chunks)} vectors for upload...")
    
    points = []
    for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i, 
                vector=vector,
                payload={ 
                    "text": chunk_text,
                    "source": source_filename,
                    "chunk_index": i
                }
            )
        )
        
    logging.info("Uploading to Qdrant (this may take a moment)...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logging.info(f"✅ Successfully uploaded {len(points)} vectors to Qdrant!")

def close_connection():
    """Explicitly closes the Qdrant connection to prevent shutdown warnings."""
    client.close()
    logging.info("Qdrant connection closed gracefully.")