# Process and upload raw data
## This code processes the metadata from AWS RDS and Markdown Files from AWS S3
## Creates a jsonl file by extracting chunks from md docs
## Jsonl file has chunks and metadeta for each chunk
## The chunks from the jsonl files are then uploaded to Qdrant database

# Installation
## Step 1: Clone the repository
```bash
git clone https://github.com/eve-rag.git

```
## Step 2: Install the libararies
Navigate into the project folder (replace `eve-rag` with the folder name if you changed it during cloning):
```bash
cd 'eve-rag'
pip install -r eve-rag/requirements.txt
```

# Usage
## Step 1: Change to the project directory
Navigate into the project folder (replace `eve-rag` with the folder name if you changed it during cloning):
```bash
cd 'eve-rag'
```

## Step 2: Review/Change parameters in config.yaml
```bash
upload_params:
  num_of_docs : 10000 #documents to be processed
  batch_size : 100 # batch size for Qdrant data upload
  vector_size : 768 # Qdrant vector size

database:
  # Qdrant collection name
  collection_name: "esa-data-aug_2"

embedding:
  model_name: "nasa-impact/nasa-smd-ibm-st-v2" # embedding model to be used for vectors 
  normalize: true 
``` 
## Step 3: Process the documents
Note: Check the number of documents you want to process in config.yaml
```bash
python eve-rag/data/process_data.py
```
## Step 4: Upload the processed chunks to Qdrant
```bash
python eve-rag/data/ulpoad_data_qdrant.py
```

