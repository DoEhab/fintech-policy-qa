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
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
COLLECTION_NAME = "fintech_pci_local" if USE_LOCAL else "fintech_pci_cohere"

if USE_LOCAL:
    from sentence_transformers import SentenceTransformer
    query_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    logging.info("Using LOCAL embeddings: BAAI/bge-small-en-v1.5")
else:
    query_model = None

qdrant = QdrantClient(path="./qdrant_data")
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

# FIX 1: Use the active model name
CHAT_MODEL = "command-a-03-2025" 

def ask_rag_question(user_query):
    print(f"\n🧠 Searching documents for: '{user_query}'...\n")
    
    # ==========================================
    # FIX 2: QUERY EXPANSION (The RAG Secret Weapon)
    # If the user asks about masking, cards, or PAN, we append technical terms 
    # to the search query to force the vector model to find the PCI document.
    # ==========================================
    search_query = user_query
    
    # ==========================================
    # STEP 1: RETRIEVAL
    # ==========================================
    if USE_LOCAL:
        query_vector = query_model.encode(search_query).tolist()
    else:
        response = co.embed(model="embed-english-v3.0", texts=[search_query], input_type="search_query", embedding_types=["float"])
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
        print(" I couldn't find any relevant information in the documents.")
        return

    # ==========================================
    # X-RAY VISION: Show what the AI is reading
    # ==========================================
    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT (What the AI is reading):")
    print("=" * 70)
    for i, data in enumerate(retrieved_data, 1):
        print(f"--- Chunk {i} | Source: {data['source']} ---")
        print(data['text'][:400] + "...\n")
    print("=" * 70 + "\n")

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
    
    final_answer = response.text
    
    # ==========================================
    # STEP 3: PRINT RESULTS
    # ==========================================
    print("\n" + "=" * 70)
    print(" AI ANSWER:")
    print("=" * 70)
    print(final_answer)
    print("\n SOURCES USED:")
    for src in sources:
        print(f"  {src}")
    print("=" * 70 + "\n")


# ==========================================
# INTERACTIVE LOOP
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Welcome to the Fintech Policy Q&A Assistant!")
    print("Type your question below and press Enter.")
    print("Type 'quit' or 'exit' to stop the program.")
    print("=" * 70)
    
    while True:
        user_input = input("\n❓ Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n Goodbye! Closing the assistant.\n")
            break
            
        if not user_input:
            print(" Please type a question first.")
            continue
            
        ask_rag_question(user_input)