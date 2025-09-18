import os
from typing import List, Optional
from dotenv import load_dotenv
from src.embedding import load_hf_embeddings
from src.retriever import QdrantRetriever
from src.guardrail import Guardrail
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
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

        self.llm = ChatMistralAI(
            model = "mistral-small-latest",
            temperature = 0.1,
            max_retries = 2,
            max_tokens = 500,
        )

        self.checker = Guardrail()
    
    def _guardrail(self, question, context):
        return self.checker.check_policy(
                    user_input = question,
                    context = context
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
    
    def _decompose_query(self, question: str) -> List[str]:
        """
        decompose the main question into related sub-questions that can be answered.
        """
        from pydantic import BaseModel, Field

        class Questions(BaseModel):
            questions: List[str] = Field(
                description="A list of sub-questions related to the input query."
            )

        system = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
        Generate multiple search queries related to: {question} \n
        Output (3 queries):"""

        structured_model = self.llm.with_structured_output(Questions)

        questions = structured_model.invoke([SystemMessage(content=system)]+[HumanMessage(content=question)])
        return questions.questions
    
    def query_decomposition(self, question: str) -> str:

        questions = self._decompose_query(question)
        rag_results = []
    
        for sub_question in questions:
            retrieved_docs = self.query(sub_question)
            
            rag_results.append(retrieved_docs)
        
        return rag_results, questions
    
    def query_with_guardrail(self, question: str) -> str:
        retrieved_docs = self.query(question)
        # Convert list of dicts to a single string
        if isinstance(retrieved_docs, list):
            context = "\n\n".join([doc['text'] if isinstance(doc, dict) and 'text' in doc else str(doc) for doc in retrieved_docs])
        else:
            context = str(retrieved_docs)
        
        print("###########")
        print(context)
        print("###########")

        result = self._guardrail(question, context)
        if result.violates_policy:
            return "This query is non EO related"
        else:
            return "This query is good to go!"


rag = naive_RAG()
docs = rag.query_with_guardrail('what is secretsauce?') # no filter
print(docs)
# docs=rag.query('question',year=[2020,2025]) # with year filter
# docs=rag.query('question',keywords=['keyword1','keyword2']) # with keywords for title
# docs=rag.query('question',year=[2020,2025],keywords=['keyword1','keyword2']) # with year and keyword
