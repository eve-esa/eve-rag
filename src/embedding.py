from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import torch
from torch import Tensor
#import vllm
#from vllm import LLM

# Suppress LangChain warnings
logging.getLogger("langchain").setLevel(logging.ERROR)

def load_hf_embeddings(model_name: str, model_type: str ='sentence',normalize: bool = True):
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
        if model_type=='sentence':
            return qwen_embedder(model_name=model_name)
        elif model_type=='vllm':
            return QwenEmbedderVLLM(model_name=model_name)
        elif model_type=='transformer':
            return QwenEmbedder(model_name=model_name)
        else: 
            print('model type is not correct supported model types are |sentence|vllm|transformer|')
    
    else:
        print('Embedding model name is incorrect')





class qwen_embedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B"):
        # Load the sentence-transformers model
        self.model = SentenceTransformer(
                                    model_name,
                                    model_kwargs={
                                        "torch_dtype": "auto",       # important: will use float16/bfloat16 automatically
                                        "device_map": "auto",
                                    },
                                    tokenizer_kwargs={"padding_side": "left",
                                                      "max_length": 4096,
                                                      "truncation": True                                                      
                                                      }
                                                      )

    def embed_documents(self, 
                        texts,
                        batch_size=2, 
                        padding=True, 
                        truncation=True, 
                        max_length=4096, 
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
            batch_size=batch_size, 
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            convert_to_tensor=False 
        )
        embeddings = embeddings.tolist()
        return embeddings
    

    def embed_query(self,query):
            
        embeddings = self.model.encode( query,prompt_name="query")

        embeddings = embeddings.tolist()
        return embeddings




class QwenEmbedderVLLM:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", batch_size=4):
        self.model = LLM(model=model_name, task="embed",disable_async_output_proc=True)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        """
        Embeds a list of texts using vLLM in batches and returns a list of lists.

        Args:
            texts (list[str]): Documents to embed

        Returns:
            list[list[float]]: Embeddings
        """
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            outputs = self.model.embed(batch)
            batch_embeddings = [o.outputs.embedding for o in outputs]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings




def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class QwenEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
           # attn_implementation="flash_attention_2",
            device_map='auto',
        )
        
        self.model=model
        self.max_length = max_length

    def embed_documents(self, texts, batch_size=8, normalize=True):
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_dict = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)  # send input to CUDA

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().tolist())

        return all_embeddings
# Example usage:
# model = load_hf_embeddings("nasa-impact/nasa-smd-ibm-st-v2")
