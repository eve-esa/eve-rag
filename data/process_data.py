from .get_rds_data import get_rds_metadata
import os
from dotenv import load_dotenv
import s3fs
from .process_chunks import process_and_write_documents
from .chunker import MarkdownTwoStepChunker
from .recursive_chunker import RecursiveMarkdownSplitter
from src.utils import *
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from tqdm import tqdm


def process_batch(batch_df, fs, chunker, output_dir, start_idx, end_idx):
    output_path = os.path.join(output_dir, f"{start_idx}_{end_idx}.jsonl")
    try:
        summary = process_and_write_documents(batch_df, fs, chunker, output_path)
        return len(batch_df), "success", summary
    except Exception as e:
        return len(batch_df), f"error: {e}", None


def main(output_dir: str, batch_size: int = 10, max_workers: int = 4):
    config = load_config("data/config.yaml")
    num_of_docs=config['chunk_params']['num_of_docs'] # num of documents to process
    chunker_type=config['chunk_params']['chunker_type']

    # Load variables from .env into the environment
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT"))
    DB_USER = os.getenv("DB_USER")
    DB_NAME = os.getenv("DB_NAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    meta_data = get_rds_metadata(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, num_of_docs=num_of_docs)

    fs = s3fs.S3FileSystem()
    os.makedirs(output_dir, exist_ok=True)


    if chunker_type=='hierarchical':
        chunker = MarkdownTwoStepChunker(max_chunk_size=1024, chunk_overlap=0,
                                     add_headers=False, merge_small_chunks=True)
    elif chunker_type=='recurcive':
        chunker = RecursiveMarkdownSplitter(chunk_size=1024, chunk_overlap=0)
    else:
        raise ValueError(f"Unknown chunker_type: {chunker_type}. "
                     f"Expected 'hierarchical' or 'recursive'.")
        

    num_batches = math.ceil(len(meta_data) / batch_size)
    batches = [meta_data.iloc[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

    results = []

    # parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_batch, batch, fs, chunker, output_dir,
                                   i*batch_size, i*batch_size+len(batch)): i
                   for i, batch in enumerate(batches)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_count, status, summary = future.result()
            if summary:
                print(f"[Batch] {os.path.basename(summary['output_file'])} | "
                      f"Docs: {summary['docs_processed']}, "
                      f"Records: {summary['records_written']}, "
                      f"Skipped: {len(summary['skipped_files'])}")
            else:
                print(f"[Batch] Error: {status}")
            results.append((batch_count, status))

    print("Processing results per batch:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and chunk documents in parallel.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save chunked documents")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Number of documents per batch")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Number of parallel threads to use")
    args = parser.parse_args()

    main(output_dir=args.output_dir, batch_size=args.batch_size, max_workers=args.max_workers)
