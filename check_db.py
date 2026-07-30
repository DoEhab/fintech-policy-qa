# check_db_health.py
from qdrant_client import QdrantClient
from collections import Counter

client = QdrantClient(path="./qdrant_data")

print("🔍 Scanning database health...\n")

# Get a large sample of records to count sources
records, next_offset = client.scroll(
    collection_name="fintech_pci_local",
    limit=10000, # Grab up to 10,000 chunks
    with_payload=["source"],
    with_vectors=False
)

print(f"📊 Total chunks actually in database: {len(records)}\n")

# Count chunks per document
source_counts = Counter()
for record in records:
    source = record.payload.get("source", "Unknown")
    source_counts[source] += 1

print("📂 Chunks per document:")
print("-" * 80)
for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {count:5d} chunks | {source}")
print("-" * 80)

if len(source_counts) < 5:
    print("\n⚠️ WARNING: Very few documents made it into the database!")