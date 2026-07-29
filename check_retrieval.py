# check_retrieval.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(path="./qdrant_data")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test query
test_query = "card masking PCI DSS"
query_vector = model.encode(test_query).tolist()

# Get top 10 results instead of 3
results = client.query_points(
    collection_name="fintech_pci_local",
    query=query_vector,
    limit=10
).points

print(f"\n🔍 Top 10 results for: '{test_query}'\n")
print("=" * 70)

for i, hit in enumerate(results, 1):
    source = hit.payload.get("source", "Unknown")
    text = hit.payload.get("text", "")[:200]  # First 200 chars
    score = hit.score
    
    print(f"\n#{i} | Score: {score:.4f}")
    print(f"📄 Source: {source}")
    print(f"📝 Text preview: {text}...")
    print("-" * 70)