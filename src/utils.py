import json
import yaml

# Format retrieved documents:
def format_docs(docs):
  doc_str = ''
  for i, doc in enumerate(docs):
    doc_str += f'Score: {doc.metadata.get("score", "None")}\n'
    doc_str += f'Document n. {i+1}\n'
    doc_str += f'TITLE: {doc.metadata.get("title", "No title")}\n' # Add title's of the paper
    if type(doc.metadata.get("headers", {})) is str:
      # Parse JSON string
      doc.metadata['headers'] = json.loads(doc.metadata.get("headers", {}))
    for key, value in doc.metadata.get("headers", {}).items():
      doc_str += f'{value}\n'
    doc_str += f'URL: {doc.metadata.get("url", "No url")}\n\n' # Add URL of the paper
    doc_str += f'{doc.page_content}\n\n'
  return doc_str


def load_config(path: str = "config.yaml") -> dict:
    """
    Loads a YAML configuration file into a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
