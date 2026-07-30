# ask_question.py
import os
import cohere
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient

try:
    from document_titles import get_document_title
except ImportError:
    def get_document_title(filename):
        return filename.replace(".pdf", "").replace("_", " ").replace("fintech-docs/", "")

load_dotenv()
logging.basicConfig(level=logging.WARNING) # Set to WARNING to keep Streamlit terminal clean

USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
COLLECTION_NAME = "fintech_pci_local" if USE_LOCAL else "fintech_pci_cohere"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")

if USE_LOCAL:
    from sentence_transformers import SentenceTransformer
    query_model = SentenceTransformer(EMBEDDING_MODEL)
    logging.info(f"Using LOCAL embeddings: {EMBEDDING_MODEL}")
else:
    query_model = None

qdrant = QdrantClient(path="./qdrant_data")
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

CHAT_MODEL = "command-a-03-2025" 

def ask_rag_question(user_query: str) -> dict:
    """
    Core RAG logic. Takes a question, returns a dictionary with 'answer' and 'sources'.
    """
    # ==========================================
    # STEP 1: RETRIEVAL
    # ==========================================
    if USE_LOCAL:
        query_vector = query_model.encode(user_query).tolist()
    else:
        response = co.embed(model="embed-english-v3.0", texts=[user_query], input_type="search_query", embedding_types=["float"])
        query_vector = response.embeddings.float[0]
    
    # Get top 5 chunks
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5 
    ).points
    
    retrieved_data = []
    sources = set()
    
    for hit in search_results:
        text = hit.payload.get("text", "")
        source = hit.payload.get("source", "Unknown")
        retrieved_data.append({"text": text, "source": get_document_title(source)})
        sources.add(get_document_title(source))
        
    if not retrieved_data:
        return {
            "answer": "I couldn't find any relevant information in the provided documents to answer that.",
            "sources": []
        }

    # ==========================================
    # STEP 2: GENERATION (LLM)
    # ==========================================
    formatted_context = "\n\n---\n\n".join([d["text"] for d in retrieved_data])
    
    system_prompt = """You are an expert AI assistant specializing in financial security and compliance policies.
Answer the user's question based on the provided context. 
Synthesize the information into a clear, direct answer. Cite the specific document names."""

    user_prompt = f"""Context:
<CONTEXT>
{formatted_context}
</CONTEXT>

Question: {user_query}"""

    response = co.chat(
        model=CHAT_MODEL,
        message=user_prompt,
        preamble=system_prompt
    )
    
    # Return the dictionary exactly as Streamlit expects it
    return {
        "answer": response.text,
        "sources": list(sources)
    }

