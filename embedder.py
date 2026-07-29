# embedder.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Read the switch from .env (defaults to False if not found)
USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

if USE_LOCAL:
    logging.info("Using LOCAL embeddings (sentence-transformers)")
    from sentence_transformers import SentenceTransformer
    # BAAI/bge-small-en-v1.5 model outputs 384 dimensions low accuracy embeddings, but is fast and free. Good for local testing.
    # Nomic/nomic-embed-text-v1.5 model outputs 768 dimensions.
    embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    
    def embed_chunks(chunks, batch_size=32):
        logging.info(f"Starting local embedding for {len(chunks)} chunks...")
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logging.info(f"Embedding local batch {i//batch_size + 1}...")
            batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True).tolist()
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

else:
    logging.info("🔵 Using COHERE API embeddings")
    import cohere
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    
    def embed_chunks(chunks, batch_size=20):
        logging.info(f"Starting Cohere embedding for {len(chunks)} chunks...")
        all_embeddings = []
        model_name = "embed-english-v3.0"
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logging.info(f"Embedding Cohere batch {i//batch_size + 1}...")
            response = co.embed(
                model=model_name, texts=batch, input_type="search_document", embedding_types=["float"]
            )
            all_embeddings.extend(response.embeddings.float)
        return all_embeddings