import os
import json
import pandas as pd
from src.utils import *

config = load_config("data/config.yaml")


def save_skipped_files(skipped_files, output_dir):
    if not skipped_files:
        return
    skipped_path = os.path.join(output_dir, "skipped_files.txt")
    with open(skipped_path, "w", encoding="utf-8") as f:
        for path in skipped_files:
            f.write(path + "\n")


def process_text(text):
    text = text.strip('""')
    text = text.replace('\\n', '\n')
    return text


def process_and_write_documents(df, fs, chunker, output_path):
    """
    Fetched markdown documents from AWS S3, processes documents, and writes all chunks to a single output file.

    Args:
        df (pd.DataFrame): DataFrame containing metadata and file paths.
        fs (fsspec.AbstractFileSystem): File system for accessing S3 files.
        chunker (Any): Object with a `.chunk(text)` method to split text into chunks.
        output_path (str): Full path (including filename) to write the JSONL output.
    
    Returns:
        dict: Summary of processing with keys:
            - "output_file" (str): Path to the output file.
            - "docs_processed" (int): Number of documents successfully processed.
            - "records_written" (int): Number of chunk records written.
            - "skipped_files" (list): List of file paths that were missing/skipped.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    skipped_files = []
    doc_count = 0
    records_written = 0
    bucket_path = config['S3_path']

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            key = row['file_path']
            s3_path = os.path.join(bucket_path, key)
            try:
                with fs.open(s3_path, 'r', encoding='utf-8') as s3_file:
                    content = process_text(s3_file.read())
                    chunks = chunker.chunk(content)
                    meta_data = row.to_dict()
                    for chunk in chunks:
                        record = {"content": chunk, "metadata": meta_data}
                        f.write(json.dumps(record) + "\n")
                        records_written += 1
                doc_count += 1
            except FileNotFoundError:
                skipped_files.append(s3_path)

    if records_written == 0 and os.path.exists(output_path):
        os.remove(output_path)

    save_skipped_files(skipped_files, os.path.dirname(output_path))

    return {
        "output_file": output_path,
        "docs_processed": doc_count,
        "records_written": records_written,
        "skipped_files": skipped_files
    }
