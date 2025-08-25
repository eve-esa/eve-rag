from qdrant_client import QdrantClient, models

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

collection_name = "esa-data-indus"

client.update_collection(
    collection_name=collection_name,
    hnsw_config=models.HnswConfigDiff(
        m=32,
        ef_construct=256,
        full_scan_threshold=10_000,
        max_indexing_threads=12,
        on_disk=True
    ),
    optimizer_config=models.OptimizersConfigDiff(
        indexing_threshold=10_000
    )
)

print(" Collection updated, indexing will run in background.")
