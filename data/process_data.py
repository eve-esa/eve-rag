from get_rds_data import get_rds_metadata
import os
from dotenv import load_dotenv
import s3fs
from process_chunks import process_and_write_documents
from chunker import MarkdownTwoStepChunker

num_of_docs=1 # num of documents to process
output_dir = "/content/drive/MyDrive/data_augmentation/" # Output directory
# Load variables from .env into the environment
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
meta_data=get_rds_metadata(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, num_of_docs=num_of_docs)

# Setup S3 filesystem
fs = s3fs.S3FileSystem()

# chunker for markdown files
chunker = MarkdownTwoStepChunker(max_chunk_size=1024, chunk_overlap=0,add_headers=False)



# Run processing (use None to process all docs)
process_and_write_documents(
    df=meta_data,
    fs=fs,
    chunker=chunker,
    output_dir=output_dir
)