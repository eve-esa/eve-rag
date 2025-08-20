#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Suppress LangChain warnings
logging.getLogger("langchain").setLevel(logging.ERROR)

def load_hf_embeddings(model_name: str, normalize: bool = True):
    """
    Loads a HuggingFaceEmbeddings model with optional normalization.

    Args:
        model_name (str): The name of the HuggingFace model to load.
        normalize (bool): Whether to normalize the embeddings. Default is True.

    Returns:
        HuggingFaceEmbeddings: The initialized embeddings model.
    """

    if model_name=="nasa-impact/nasa-smd-ibm-st-v2":
        encode_kwargs = {"normalize_embeddings": normalize}
        return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
    
    elif model_name=="Qwen/Qwen3-Embedding-4B":
        return qwen_embedder(model_name=model_name)
    
    else:
        print('Embedding model name is incorrect')





class qwen_embedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B"):
        # Load the sentence-transformers model
        #self.model = SentenceTransformer(model_name)
        self.model = SentenceTransformer(
                                    model_name,
                                    model_kwargs={
                                        "torch_dtype": "auto",       # important: will use float16/bfloat16 automatically
                                        "device_map": "auto",
                                    },
                                    tokenizer_kwargs={"padding_side": "left"},
                                )

    def embed_documents(self, texts, 
                        padding=True, 
                        truncation=True, 
                        max_length=2048, 
                        normalize=True):
        """
        Encodes a list of texts into embeddings.

        Args:
            texts (list[str]): Documents to embed
            padding (bool/str): True = dynamic padding, 'max_length' = fixed length
            truncation (bool): Whether to truncate texts beyond max_length
            max_length (int): Max tokens allowed
            normalize (bool): Whether to L2 normalize embeddings

        Returns:
            np.ndarray: Embeddings array (num_texts x embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            convert_to_tensor=False 
        )
        embeddings = embeddings.tolist()
        return embeddings

# Example usage:
# model = load_hf_embeddings("nasa-impact/nasa-smd-ibm-st-v2")
