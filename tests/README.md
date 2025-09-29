# IR Test Scripts

This folder contains test scripts for evaluating different components of the **Information Retrieval (IR) pipeline**.  
The tests cover retrieval accuracy, reranking, and Q/A dataset generation.


##  Files Overview

### 1. `single_chunk_test.ipynb`
- **Purpose**: Tests retrieval with **single chunks**.
- **Scenarios**:
  - Direct chunk matching
  - Paraphrased chunk matching
- **Usage**: Run this notebook to validate if the retriever correctly handles exact and paraphrased queries against single chunks.

### 2. `multi_chunk_test.ipynb`
- **Purpose**: Tests retrieval with **multiple chunks**.
- **Scenarios**:
  - Two chunks from the **same document**
  - Two chunks from **different documents**
- **Usage**: Run this notebook to check retrieval robustness when multiple related or unrelated chunks are tested for retrieval.


### 3. `QA_dataset_generation.ipynb`
- **Purpose**: Generates a **Q/A dataset** using an LLM.
- **Input**: ~2,000 documents
- **Output**: Synthetic Q/A pairs for retrieval benchmarking
- **Usage**: Run this notebook to generate the dataset for evaluation.


### 4. `QA_retrieval_test.ipynb`
- **Purpose**: Tests **retrieval performance** on generated Q/A datasets.
- **Usage**: Run this after generating the dataset (`QA_dataset_generation.ipynb`) to validate retrieval quality.


### 5. `rearranking_test.ipynb`
- **Purpose**: Tests **re-ranking** of retrieved results using an LLM.
- **Usage**: Run this notebook to evaluate how reranking improves retrieval accuracy.



##  Results Tracking
All test results are tracked in the following Google Sheet:

[Test Results Spreadsheet](https://docs.google.com/spreadsheets/d/1zo2c2VKkxVilBUGJkULpxqXtht7TqEyq43ePfxhSnRw/edit?gid=0#gid=0)

