from typing import List
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct
from langchain_core.retrievers import BaseRetriever


class QdrantRetriever:
    """
    Retrieves the top-k relevant documents from Qdrant for the given query.

    Args:
        query (str): The query string.

    Returns:
        List[Document]: A list of LangChain Document objects with text Chunks and metadata.
    

    Args:
        embedding: An embedding model instance with `embed_query()` method
        api_key (str): API key for Qdrant Cloud
        qdrant_url (str): URL of the Qdrant service
        collection_name (str): Name of the Qdrant collection to search
        k (int): Number of top results
    """

    def __init__(self, embedding, api_key: str, qdrant_url: str,
                 collection_name: str = 'esa-nasa-workshop', k: int = 3):
        self._client = QdrantClient(url=qdrant_url, api_key=api_key)
        self.embedding = embedding
        self.collection_name = collection_name
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:

        query_emb = self.embedding.embed_query(query)

        search_result = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_emb,
            limit=self.k,
        )

        docs = []
        for hit in search_result:
            payload = hit.payload or {}
            content = payload.get("text", "")
            metadata = {k: v for k, v in payload.items() if k != "text"}
            metadata["score"] = hit.score
            docs.append(Document(page_content=content, metadata=metadata))

        return docs

# Example usage:
# retriever = QdrantRetriever(embedding, api_key, qdrant_url)
# docs = retriever.get_relevant_documents("What is the role of Earth in the Solar System?")
