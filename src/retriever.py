from typing import List, Optional
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchText
import time

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchText
import time
from typing import List, Optional
def get_filter(year: Optional[List[int]] = None, keywords: Optional[List[str]] = None) -> Optional[Filter]:
    """
    Create a Qdrant filter based on optional year range and list of keywords.

    Args:
        year (Optional[List[int]]): A list containing two integers [start_year, end_year] for filtering by year.
        keywords (Optional[List[str]]): A list of keyword strings to filter the "title" field.

    Returns:
        Optional[Filter]: A Qdrant Filter object combining year and keyword conditions, or None if no filter is applied.
    """
    conditions = []
    # Skip year filter if input is invalid
    if year and len(year) == 2 and all(isinstance(y, (int, float)) for y in year):
        conditions.append(
            FieldCondition(
                key="year",
                range=Range(gte=int(year[0]), lte=int(year[1]))
            )
        )

    if keywords:
        keyword_conditions = [
            FieldCondition(
                key="title",
                match=MatchText(text=kw)
            ) for kw in keywords
        ]
        if len(keyword_conditions) == 1:
            conditions.append(keyword_conditions[0])
        else:
            return Filter(
                must=conditions,
                should=keyword_conditions
            )

    return Filter(must=conditions) if conditions else Filter()


class QdrantRetriever:
    """
    Retrieves the top-k relevant documents from Qdrant for the given query.
    """

    def __init__(self, embedding, api_key: str, qdrant_url: str,
                 collection_name: str = 'esa-data-aug_2', k: int = 3):
        self._client = QdrantClient(url=qdrant_url, api_key=api_key)
        self.embedding = embedding
        self.collection_name = collection_name
        self.k = k

    def get_relevant_documents(self, query: str, year: List[int] = None, keywords: List[str] = None) -> List[Document]:
        """
        Retrieves top-k relevant documents from Qdrant based on query and optional filters.

        Args:
            query (str): The query string to search.
            year (List[int], optional): A list with two values [start_year, end_year] to filter by publication year.
            keywords (List[str], optional): List of keywords to filter by title.

        Returns:
            List[Document]: A list of LangChain Document objects containing the matched text and metadata.
        """
        query_emb = self.embedding.embed_query(query)
        query_filter = get_filter(year=year, keywords=keywords)

        start_time = time.time()
        try:
            search_result = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_emb,
                limit=self.k,
                query_filter=query_filter
            )
        except Exception as e:
            print(f"Search failed: {e}")
            search_result = []
        end_time = time.time()
        print(f'Search time : {(end_time-start_time)*1000} ms')
        docs = []
        for hit in search_result:
            payload = hit.payload or {}
            content = payload.get("content", "")
            metadata = {k: v for k, v in payload.items() if k != "text"}
            metadata["score"] = hit.score
            docs.append(Document(page_content=content, metadata=metadata))
        
        return docs

"""
retriever = QdrantRetriever(embedding, api_key, qdrant_url,collection_name,k)

# No filters
docs = retriever.get_relevant_documents("What is the role of Earth in the Solar System?")

# Year filter only
docs = retriever.get_relevant_documents("What is the role of Earth in the Solar System?", year=[2010, 2020])

# Keywords filter only
docs = retriever.get_relevant_documents("What is the role of Earth in the Solar System?", keywords=["soil", "crop", "plant"])

# Both filters
docs = retriever.get_relevant_documents("What is the role of Earth in the Solar System?", year=[2000, 2020], keywords=["soil", "crop"])

"""