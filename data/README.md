# Process raw data and upload to Qdrant
This code:

- Processes the metadata from AWS RDS
- Reads Markdown files from AWS S3
- Extracts chunks from the markdown documents
- Creates a `.jsonl` file containing:
  - Each chunk of content
  - Associated metadata for each chunk
- Uploads the chunks from the `.jsonl` file to a Qdrant vector database

# Installation
## Step 1: Clone the repository
```bash
git clone https://github.com/eve-rag.git

```
## Step 2: Install the libararies
Navigate into the project folder (replace `eve-rag` with the folder name if you changed it during cloning):
```bash
cd 'eve-rag'
pip install -r requirements.txt
```

# Usage
## Step 1: Change to the project directory
Navigate into the project folder (replace `eve-rag` with the folder name if you changed it during cloning):
```bash
cd 'eve-rag'
```

## Step 2: Review/Change parameters in config.yaml
```
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
python data/process_data.py
```
## Step 4: Upload the processed chunks to Qdrant
```bash
python data/ulpoad_data_qdrant.py
```

