from qdrant_client import QdrantClient, models
import os

QDRANT_URL=os.getenv('QDRANT_URL')
QDRANT_API_KEY=os.getenv('QDRANT_API_KEY')

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

collection_name = "esa-data-indus"


client.update_collection(
    collection_name=collection_name,
    hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=128,
        full_scan_threshold=10_000,
        max_indexing_threads=2,
        on_disk=True
    ),
    optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,          # start indexing after 20k vectors per segment
            memmap_threshold=5000,             # smaller than indexing_threshold; helps with RAM limits
            deleted_threshold=0.2,             # when >20% deleted, trigger segment cleanup
            vacuum_min_vector_number=1000,     # minimum segment size for vacuuming
            default_segment_number=2,          # spread data across 2 segments
            max_segment_size=6_000_000,       # keep segments smaller (avoid huge merges) but larger for lower latency
            max_optimization_threads=1,        # limit parallel merges (less memory/disk pressure)
        ),
)

print(" Collection updated, indexing will run in background.")
