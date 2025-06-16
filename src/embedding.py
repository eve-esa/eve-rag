from langchain_community.embeddings import HuggingFaceEmbeddings

def load_hf_embeddings(model_name: str, normalize: bool = True):
    """
    Loads a HuggingFaceEmbeddings model with optional normalization.

    Args:
        model_name (str): The name of the HuggingFace model to load.
        normalize (bool): Whether to normalize the embeddings. Default is True.

    Returns:
        HuggingFaceEmbeddings: The initialized embeddings model.
    """
    encode_kwargs = {"normalize_embeddings": normalize}
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)

# Example usage:
# model = load_hf_embeddings("nasa-impact/nasa-smd-ibm-st-v2")
