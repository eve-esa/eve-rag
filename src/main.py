import os
from typing import List, Optional
from dotenv import load_dotenv
from src.embedding import load_hf_embeddings
from src.retriever import QdrantRetriever
from langchain_core.documents import Document
from src.utils import *

class naive_RAG:
    """
    A reusable engine to query Qdrant with embedding-based search and optional filters.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        # Load environment and config
        load_dotenv()

        config = load_config(config_path)

        if config["cluster"]=='llm4eo':
            print('cluster llm4eo loaded')
            QDRANT_API_KEY = os.getenv("QDRANT_API_KEY_1")
            QDRANT_URL = os.getenv("QDRANT_URL_1")
        else:
            print('cluster eve-collections loaded')
            QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
            QDRANT_URL = os.getenv("QDRANT_URL")

        # Load embedding model once
        embedding_cfg = config["embedding"]
        embedding_model = load_hf_embeddings(
            model_name=embedding_cfg["model_name"],
            normalize=embedding_cfg.get("normalize", True)
        )

        # Initialize Qdrant retriever once
        self.retriever = QdrantRetriever(
            embedding=embedding_model,
            api_key=QDRANT_API_KEY,
            qdrant_url=QDRANT_URL,
            collection_name=config["database"]["collection_name"],
            k=config["database"]["top_k"]
        )

    def query(self, question: str, year: Optional[List[int]] = None, keywords: Optional[List[str]] = None,ret_time:bool=False) -> str:
        """
        Query Qdrant for relevant documents.

        Args:
            question (str): The input query string.
            year (List[int], optional): A list with [start_year, end_year] to filter by year.
            keywords (List[str], optional): List of keywords to filter by title.

        Returns:
            List[dict]: List of documents as dictionaries with text and metadata.
        """
        if ret_time:
            docs,ret_time = self.retriever.get_relevant_documents(
                query=question,
                year=year,
                keywords=keywords,
                ret_time=ret_time
            )
            docs=format_docs(docs)
            #print(docs)
            return docs
        else:
            docs = self.retriever.get_relevant_documents(
                query=question,
                year=year,
                keywords=keywords,
                ret_time=ret_time
            )
            docs=format_docs(docs)
            #print(docs)
            return docs
    

"""
Usage:

rag=naive_RAG()
docs=rag.query('question') # no filter
docs=rag.query('question',year=[2020,2025]) # with year filter
docs=rag.query('question',keywords=['keyword1','keyword2']) # with keywords for title
docs=rag.query('question',year=[2020,2025],keywords=['keyword1','keyword2']) # with year and keyword

"""
