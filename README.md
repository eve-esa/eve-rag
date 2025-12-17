# EVE-RAG
This repository contains code and experiments for a Retrieval-Augmented Generation (RAG) pipeline. The goal is to explore techniques to improve question answering and reasoning by augmenting LLMs with external knowledge sources.

Specifically you will find a notebook section demonstrating the different components of the RAG pipeline that has been built for EVE.
You will also find a scripts section with some utility scripts that covers some time-bound operations if you want to run them outside a notebook environment.

## Notebooks

The notebooks provide an interactive walkthrough of the complete RAG pipeline, from data preparation to advanced optimization techniques:

### 1. Chunking Strategies (`01_chunking_strategies.ipynb`)
Explores text chunking techniques for splitting large documents into manageable pieces:
- **Recursive Chunking**: Hierarchical splitting by paragraphs → sentences → words while preserving structure
- **Hierarchical Chunking**: Section-based splitting using Markdown headers with special handling for LaTeX formulas and tables
- Demonstrates how chunk size and overlap affect retrieval quality
- Preserves formatting and semantic meaning throughout the chunking process

### 2. Data Ingestion (`02_ingest.ipynb`)
Implements the complete data ingestion pipeline:
- **Loading**: Process and load data in text format
- **Embeddings**: Generate vector representations using domain-specific models (NASA-SMD-IBM-ST-V2)
- **Vector Store**: Index and store embeddings in Qdrant vector database
- Covers best practices for embedding model selection for Earth Observation documents
- Demonstrates batch processing and efficient uploading to Qdrant

### 3. Retrieval and Generation (`03_RAG.ipynb`)
Builds a complete end-to-end RAG pipeline:
- **Retrieval**: Query the vector database to find relevant document chunks
- **Generation**: Use retrieved context to generate accurate, grounded answers with EVE LLM
- Demonstrates the full workflow from user query to final response
- Shows how to combine embedding-based retrieval with language model generation

### 4. Evaluation (`04_evaluation.ipynb`)
Comprehensive evaluation framework for RAG systems:
- **Token-Level Metrics**: IoU, Precision, Recall, F1 (fine-grained text overlap)
- **Passage-Level Metrics**: Coverage, Accuracy, Precision, Recall, F1 (reference passage retrieval)
- **Document-Level Metrics**: Coverage, Precision (source document retrieval)
- Generates synthetic Q&A datasets with validated references
- Evaluates performance at different K values (top-3, top-5, top-10, top-15)
- Provides detailed analysis and comparison tables

### 5. Query Rewriting (`05_query_rewriting.ipynb`)
Demonstrates query rewriting for multi-turn conversational RAG:
- **Problem**: Context-dependent queries with pronouns and implicit references
- **Solution**: Transform queries into self-contained versions using conversation history
- Handles pronoun resolution, abbreviation expansion, and context incorporation
- Shows significant improvement in retrieval quality for follow-up questions
- Includes before/after comparisons with retrieval score analysis

### 6. Re-ranking (`06_reranking.ipynb`)
Advanced re-ranking techniques to improve retrieval quality:
- **Two-Stage Approach**: Fast retrieval (embedding-based) → Precise re-ranking (cross-encoder)
- **NASA-SMD-IBM-Ranker**: Domain-specific re-ranking model for scientific/EO documents
- **Metrics**: Mean Reciprocal Rank (MRR) and rank distribution analysis
- Demonstrates position changes (original rank → new rank) with visual indicators
- Shows trade-offs between accuracy gains and computational cost
- Provides comprehensive comparison: with vs without re-ranking

## Scripts

Utility scripts for production-ready operations and large-scale processing:

### 1. Data Processing (`process_chunks.py`)
Processes and chunks documents from AWS S3:
- Fetches markdown documents from S3 buckets
- Applies chunking strategies to split documents
- Writes processed chunks to JSONL format
- Handles errors gracefully and logs skipped files
- Useful for batch processing large document collections

### 2. Qdrant Index Creation (`create_file_path_index.py`)
Creates keyword indexes in Qdrant for efficient filtering:
- Creates a keyword index for the `file_path` field
- Enables fast filtering by document source
- Required before running payload update operations
- Usage: `python create_file_path_index.py --config config.yaml`

### 3. Payload Updates (`update_qdrant_payload.py`)
Updates metadata in Qdrant vector database:
- Updates payload fields for points matching specific file paths
- Preserves existing fields not included in updates
- Supports dry-run mode for safe testing
- Processes updates in batches for efficiency
- Usage: `python update_qdrant_payload.py --config config.yaml --dry-run`

### 4. Q&A Dataset Generation (`generate_qa_dataset.py`)
Generates evaluation datasets using LLMs:
- Creates question-answer pairs from documents
- Validates that references exist in Qdrant chunks
- Supports both local and API-based LLMs (OpenAI, etc.)
- Maps references to chunk numbers for evaluation
- Outputs validated Q&A dataset in JSON format
- Usage: `python generate_qa_dataset.py --config config.yaml`

### 5. RAG Evaluation (`evaluate_rag.py`)
Evaluates RAG system performance using Q&A datasets with ground truth:
- **Token-Level Metrics**: IoU, Precision, Recall, F1 (fine-grained text overlap)
- **Passage-Level Metrics**: Coverage, Accuracy, Precision, Recall, F1 (reference passage retrieval)
- **Document-Level Metrics**: Coverage, Accuracy, Precision, Recall (source document retrieval)
- Supports multiple K values for top-K retrieval analysis (e.g., top-3, top-5, top-10, top-15)
- Fuzzy matching with configurable threshold for robust reference detection
- Outputs detailed per-question results and summary statistics
- Tracks embedding and retrieval latency
- Usage: `python evaluate_rag.py --config config.yaml`


## Funding

This project is supported by the European Space Agency (ESA) Φ-lab through the Large Language Model for Earth Observation and Earth Science project, as part of the Foresight Element within FutureEO Block 4 programme.

## Citation

If you use this project in academic or research settings, please cite:

## License

This project is released under the Apache 2.0 License - see the [LICENSE](LICENSE) file for more details.



