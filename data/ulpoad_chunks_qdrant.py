from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, HnswConfigDiff, OptimizersConfigDiff,PointStruct
from qdrant_client import models
from src.embedding import load_hf_embeddings
import os
import json
import time
import hashlib
from tqdm import tqdm
from dotenv import load_dotenv
from src.utils import *


def create_qdrant_collection(QDRANT_URL,QDRANT_API_KEY,collection_name:str,vector_size:int):
    #Create a Qdrant collection if not already present.

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if client.collection_exists(collection_name):
        print("Collection already exists")
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                on_disk=True,
            ),
            shard_number=4,  # Increase shards for large data
            replication_factor=1,
            on_disk_payload=True,
        )

        client.update_collection(
            collection_name=collection_name,
            hnsw_config=HnswConfigDiff(
                on_disk=True, # saves indexing graph on disk
                m=48 # keep it low if low memory available

            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=100000,
                memmap_threshold=50000, # should be lower than indexing_threshold when RAM is limited
                deleted_threshold=0.1, 
            ),
        )
        client.create_payload_index(
        collection_name=collection_name,
        field_name="title",
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=50,
            lowercase=True,
        ))

        # Add integer index for 'year'
        client.create_payload_index(
            collection_name=collection_name,
            field_name="year",
            field_schema="integer"
        )

        print("Collection created and optimized.")


def string_to_uint(s: str) -> int:

    hash_bytes = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)

def get_existing_ids(client, collection_name):
    #get existing ids so there are no duplicates"
    """
    args:
    collection_name(string): name of collection in Qdrant
    client: Qdrant client

    return:
    ids(list) : list of Ids of the existing points/chunks in Qdrant
    """
    existing_ids = set()
    scroll_offset = None

    while True:
        response = client.scroll(
            collection_name=collection_name,
            offset=scroll_offset,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        for point in response[0]:
            existing_ids.add(point.id)
        if response[1] is None:
            break
        scroll_offset = response[1]

    return existing_ids

def upload_batch(batch_ids, batch_chunks, batch_metadata, client, collection_name, embedder):
    """ uploads batches of the chuncks to the Qdrant collection

    args:
    batch_ids(list): uniques ids of chunks
    batch_chunks(list): list of chunks of text
    batch_metadata(list): metadata assosiated to the chunks
    client: Qdrant client
    collection_name(string): name of collection in Qdrant
    embedder: Embedding model to convert chunks to vector embeddings
    
    """
    try:
        batch_vectors = embedder.embed_documents(batch_chunks)
    except Exception as e:
        print(f"Embedding error: {e}")
        return

    points = [
        PointStruct(id=id, vector=vec, payload=meta)
        for id, vec, meta in zip(batch_ids, batch_vectors, batch_metadata)
    ]

    for attempt in range(3):
        try:
            client.upload_collection(collection_name=collection_name,
                                     ids=batch_ids,
                                     vectors=batch_vectors,
                                     payload=batch_metadata,
                                     parallel=10)
            return
        except Exception as e:
            print(f"Error uploading batch: {e}")
            time.sleep(10)
            if attempt < 2:
                print("Retrying...")
            else:
                print("Skipping batch.")

def batch_upload_chunks(ids, QDRANT_URL,QDRANT_API_KEY, collection_name, chunks, metadata, embedder, batch_size=100):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    uint_ids = [string_to_uint(id) for id in ids]
    to_process = list(zip(uint_ids, chunks, metadata, ids))  # Keep original ID for debug

    existing_ids = get_existing_ids(client, collection_name)
    to_process = [item for item in to_process if item[0] not in existing_ids]

    print(f"Skipping {len(uint_ids) - len(to_process)} existing IDs")
    print(f"Uploading {len(to_process)} new vectors")

    for i in tqdm(range(0, len(to_process), batch_size), desc="Uploading to Qdrant"):
        batch = to_process[i:i+batch_size]
        batch_ids = [item[0] for item in batch]
        batch_chunks = [item[1] for item in batch]
        batch_metadata = [item[2] for item in batch]

        upload_batch(batch_ids,batch_chunks,batch_metadata,client,collection_name,embedder)



def read_chucks_from_json(file_path):

    chunks, metadata, ids = [], [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            chunk = doc["content"]
            meta = doc["metadata"]

            if "year" in meta:
                try:
                    meta["year"] = int(float(meta["year"]))
                except:
                    meta["year"] = None

            if "title" in meta:
                meta["title"] = str(meta["title"])
            
            if "content" not in meta:
                meta["content"] = chunk

            ids.append(f"{os.path.basename(file_path)}_{i}")
            chunks.append(chunk)
            metadata.append(meta)
    
    return ids,chunks,metadata



def main():
    load_dotenv()
    config = load_config("data/config.yaml")


    batch_size = config['upload_params']['batch_size'] 
    vector_size = config['upload_params']['vector_size']
    num_of_docs=config['upload_params']['num_of_docs']
    collection_name=config["database"]["collection_name"]

    current_dir = os.getcwd() 
    output_dir = os.path.join(current_dir, "chunked_data")
    file_path = os.path.join(output_dir, f"{num_of_docs}_docs.jsonl")# path for the jsonl file of the chuncks

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")


    # Load embedding model
    embedding_cfg = config["embedding"]
    embedding_model = load_hf_embeddings(
        model_name=embedding_cfg["model_name"],
        normalize=embedding_cfg.get("normalize", True)
    )

    create_qdrant_collection(QDRANT_URL,QDRANT_API_KEY,collection_name,vector_size)
    ids,chunks,metadata=read_chucks_from_json(file_path)
    batch_upload_chunks(ids=ids, 
                        QDRANT_URL=QDRANT_URL,
                        QDRANT_API_KEY=QDRANT_API_KEY, 
                        collection_name=collection_name, 
                        chunks=chunks,
                        metadata=metadata, 
                        embedder=embedding_model,
                        batch_size=batch_size)



if __name__ == "__main__":
    main()

