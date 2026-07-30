# list_sources.py
from qdrant_client import QdrantClient
from collections import Counter

# 1. Connect to your local database
client = QdrantClient(path="./qdrant_data")
COLLECTION_NAME = "fintech_pci_local"

print(f"🔍 Scanning collection: '{COLLECTION_NAME}'...\n")

try:
    # 2. Get the total number of chunks (points) in the database
    collection_info = client.get_collection(COLLECTION_NAME)
    total_chunks = collection_info.points_count
    print(f"📊 TOTAL CHUNKS IN DATABASE: {total_chunks:,}\n")

    # 3. Fetch the payloads (metadata) of the chunks
    # We set limit to 10,000 which is more than enough for your current dataset
    records, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=["source"], # We only need the 'source' field, saves memory
        with_vectors=False       # We don't need the actual embedding vectors
    )

    # 4. Count how many chunks belong to each document
    source_counts = Counter()
    for record in records:
        source = record.payload.get("source", "Unknown_Source")
        source_counts[source] += 1

    # 5. Print the results nicely
    print("=" * 80)
    print(f"📂 UNIQUE DOCUMENTS FOUND: {len(source_counts)}")
    print("=" * 80)
    
    # Sort by chunk count (highest first)
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        # Clean up the filename for better readability
        clean_name = source.replace("fintech-docs/", "").replace(".pdf", "")
        print(f"  {count:4d} chunks | {clean_name}")
        
    print("=" * 80)

except Exception as e:
    print(f"❌ Error reading database: {e}")
    print("Make sure you have run `python main.py` to create the database first.")