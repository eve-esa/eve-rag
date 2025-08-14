from .get_rds_data import get_rds_metadata
import os
from dotenv import load_dotenv
import s3fs
from .process_chunks import process_and_write_documents
from .chunker import MarkdownTwoStepChunker
from src.utils import *
import argparse

def main(output_dir: str = None):

    config = load_config("data/config.yaml")
    num_of_docs=config['upload_params']['num_of_docs'] # num of documents to process
    # Load variables from .env into the environment
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT"))
    DB_USER = os.getenv("DB_USER")
    DB_NAME = os.getenv("DB_NAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    meta_data=get_rds_metadata(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, num_of_docs=num_of_docs)

    # Setup S3 filesystem
    fs = s3fs.S3FileSystem()

    # chunker for markdown files
    os.makedirs("logs", exist_ok=True)
    chunker = MarkdownTwoStepChunker(max_chunk_size=1024, chunk_overlap=0,add_headers=False)



    # Run processing (use None to process all docs)
    process_and_write_documents(
        df=meta_data,
        fs=fs,
        chunker=chunker,
        output_dir=output_dir,
        total_docs=num_of_docs
    )


if __name__ == "__main__":
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "chunked_data")

    parser = argparse.ArgumentParser(description="Processed chunk documents.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="Directory to save chunked documents"
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)