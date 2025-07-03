# EVE-RAG
This repository contains code and experiments for a Retrieval-Augmented Generation (RAG) pipeline. The goal is to explore techniques to improve question answering and reasoning by augmenting LLMs with external knowledge sources.

## Resources
- Rag Survey 2023: https://arxiv.org/abs/2312.10997 
- Rag Survey 2024: https://arxiv.org/abs/2402.19473 
- Rag survey github: https://github.com/Tongji-KGLLM/RAG-Survey
- Rag model first proposed by Facebook: https://arxiv.org/pdf/2005.11401.pdf
- PaperQA - how to do high-accuracy RAG with scientific papers: https://github.com/Future-House/paper-qa

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/eve-rag.git
cd eve-rag
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
Make sure to create a .env file in the root directory with your Qdrant API credentials:
```bash .env
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_URL=https://your-qdrant-url
``` 
## Configration
Configuration parameters such as **collection name**, **embedding model**, and **top-k** results are defined in **src/config.yaml**
```bash
database:
  # Qdrant or other DB config
  collection_name: "esa-data-aug_2"
  top_k: 5 # number of chunks with highest matches

embedding:
  model_name: "nasa-impact/nasa-smd-ibm-st-v2"
  normalize: true
```
## Usage
Initialize and use the naive_RAG class in your Python script.
```bash
from src.main import naive_RAG
rag = naive_RAG()
```
##### Query
```bash
docs = rag.query("What is the role of Earth in the Solar System?")
```
##### Query with year filter:
```bash 
docs = rag.query("What missions launched recently?", year=[2015, 2023])
```
##### Query with keyword filter (searches in title):
```bash
docs = rag.query("Explain satellite imaging", keywords=["satellite", "remote sensing"])
```
##### Query with both year and keyword filters:
```bash
docs = rag.query("Explain satellite imaging techniques",year=[2010, 2020],keywords=["satellite", "sensor"])
```