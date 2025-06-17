from retriever import QdrantRetriever
from utils import *
from load_dataset import load_data
from embedding import load_hf_embeddings
import random
import os
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()
api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
# Load config
config = load_config("src/config.yaml")

# Load dataset
test_cfg = config["test_config"]
dataset = load_data(test_cfg["dataset"],split=test_cfg["split"])


# Load embedding model
embedding_cfg = config["embedding"]
embedding_model = load_hf_embeddings(
    model_name=embedding_cfg["model_name"],
    normalize=embedding_cfg.get("normalize", True)
)

# Load retriever
retriever = QdrantRetriever(
    embedding=embedding_model,
    api_key=api_key,  
    qdrant_url=qdrant_url,
    collection_name=config["database"]["collection_name"],
    k=5
)


# Get a sample question
idx = random.randint(0, len(dataset))
sample = dataset[idx]
question = sample[test_cfg["question_column"]]

# Retrieve and print
docs = retriever.get_relevant_documents(question)
print("Question:", question)
print(format_docs(docs))
