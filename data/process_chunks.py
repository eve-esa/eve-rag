import os
import json
import pandas as pd
from tqdm import tqdm


def get_output_file_path(output_dir, total_docs):
    """
    Returns a fixed output file path.
    """
    return os.path.join(output_dir, f"{total_docs}_docs.jsonl")


def save_skipped_files(skipped_files, output_dir):
    """
    Saves list of skipped files to a .txt file.
    """
    if not skipped_files:
        return

    skipped_path = os.path.join(output_dir, "skipped_files.txt")
    with open(skipped_path, "w", encoding="utf-8") as f:
        for path in skipped_files:
            f.write(path + "\n")

def process_text(text):
    # Remove the starting and ending quotes
    text = text.strip('""')

    # Replace the escaped "\n" with actual newline characters
    text = text.replace('\\n', '\n')
    return text


def process_and_write_documents(df, fs, chunker, output_dir,total_docs):
    """
    Fetched markdown documents from aws s3 , processes documents and writes all chunks to a single output file.

    Args:
        df (pd.DataFrame): DataFrame containing metadata and file paths.
        fs (fsspec.AbstractFileSystem): File system for accessing data files.
        chunker (Any): Object with `.chunk(text)` method.
        output_dir (str): Directory to write output.
        max_docs_to_process (int or None): Max number of documents to process. None means all.
    """
    os.makedirs(output_dir, exist_ok=True)

    skipped_files = []
    doc_count = 0
    records_written = 0

    output_path = get_output_file_path(output_dir, total_docs)
    output_file = None

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):

            key = row['file_path']
            s3_path = f's3://llm4eo-s3/raw_data_dedup_cleaned/{key}'

            try:
                with fs.open(s3_path, 'r', encoding='utf-8') as s3_file:
                    content = process_text(s3_file.read())
                    chunks = chunker.chunk(content)
                    meta_data = row.to_dict()

                    # Open file only if there's something to write
                    if output_file is None:
                        output_file = open(output_path, "w", encoding="utf-8")

                    for chunk in chunks:
                        record = {
                            "content": chunk.page_content,
                            "metadata": {
                                **meta_data,
                                "header": chunk.metadata
                            }
                        }
                        output_file.write(json.dumps(record) + "\n")
                        records_written += 1

                doc_count += 1

            except FileNotFoundError:
                print(f"Skipping missing file: {s3_path}")
                skipped_files.append(s3_path)

    finally:
        if output_file:
            output_file.close()

        save_skipped_files(skipped_files, output_dir)

        print(f"\nFinished. Documents processed: {doc_count}, Records written: {records_written}")
        print(f"Skipped files: {len(skipped_files)}")
        if not records_written and os.path.exists(output_path):
            os.remove(output_path)
            print(f"Removed empty output file: {output_path}")
