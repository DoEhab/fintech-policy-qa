# check_db.py
from qdrant_client import QdrantClient

# 1. Connect to the local database folder
client = QdrantClient(path="./qdrant_data")
COLLECTION_NAME = "fintech_pci_docs"

print(f"🔍 Inspecting collection: {COLLECTION_NAME}\n")

# 2. Get high-level stats
info = client.get_collection(COLLECTION_NAME)

# FIX: Changed vectors_count to points_count for newer Qdrant versions
print(f"📊 Total chunks (points) stored: {info.points_count}\n")

# 3. Retrieve a sample of the actual data
records, next_offset = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=3,               # Grab 3 chunks to verify
    with_payload=True,     # YES: We want to see the text and metadata
    with_vectors=False     # NO: Hide the 1024 numbers, we just want the text
)

print("📄 Sample Data Retrieved from Database:")
print("=" * 60)
for record in records:
    print(f"🆔 Point ID: {record.id}")
    print(f"📁 Source: {record.payload.get('source')}")
    
    # Print the first 200 characters of the text
    text_snippet = record.payload.get('text', '')
    print(f"📝 Text: {text_snippet[:200]}...\n")
    print("-" * 60)