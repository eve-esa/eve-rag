import os
from typing import List, Optional
from dotenv import load_dotenv
from utils import load_config
from embedding import load_hf_embeddings
from retriever import QdrantRetriever
from langchain_core.documents import Document
from utils import *

class naive_RAG:
    """
    A reusable engine to query Qdrant with embedding-based search and optional filters.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        # Load environment and config
        load_dotenv()
        api_key = os.getenv("QDRANT_API_KEY")
        qdrant_url = os.getenv("QDRANT_URL")
        config = load_config(config_path)

        # Load embedding model once
        embedding_cfg = config["embedding"]
        embedding_model = load_hf_embeddings(
            model_name=embedding_cfg["model_name"],
            normalize=embedding_cfg.get("normalize", True)
        )

        # Initialize Qdrant retriever once
        self.retriever = QdrantRetriever(
            embedding=embedding_model,
            api_key=api_key,
            qdrant_url=qdrant_url,
            collection_name=config["database"]["collection_name"],
            k=config["database"]["top_k"]
        )

    def query(self, question: str, year: Optional[List[int]] = None, keywords: Optional[List[str]] = None) -> str:
        """
        Query Qdrant for relevant documents.

        Args:
            question (str): The input query string.
            year (List[int], optional): A list with [start_year, end_year] to filter by year.
            keywords (List[str], optional): List of keywords to filter by title.

        Returns:
            List[dict]: List of documents as dictionaries with text and metadata.
        """
        docs = self.retriever.get_relevant_documents(
            query=question,
            year=year,
            keywords=keywords
        )
        docs=format_docs(docs)
        print(docs)
        return docs
    

"""
Usage:

rag=naive_RAG()
docs=rag.query('question') # no filter
docs=rag.query('question',year=[2020,2025]) # with year filter
docs=rag.query('question',keywords=['keyword1','keyword2']) # with keywords for title
docs=rag.query('question',year=[2020,2025],keywords=['keyword1','keyword2']) # with year and keyword

"""
