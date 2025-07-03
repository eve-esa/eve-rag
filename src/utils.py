import json
import yaml

# Format retrieved documents:
def format_docs(docs):
  doc_str = ''
  for i, doc in enumerate(docs):
    doc_str += f'Score: {doc.metadata.get("score", "None")}\n'
    doc_str += f'Document n. {i+1}\n'
    doc_str += f'TITLE: {doc.metadata.get("title", "No title")}\n' # Add title's of the paper
    if (doc.metadata.get("header", {})) is str:

      # Parse JSON string
      doc.metadata['header'] = json.loads(doc.metadata.get("header", {}))
    for key, value in doc.metadata.get("header", {}).items():
      doc_str += f'{value}\n'
    doc_str += f'URL: {doc.metadata.get("url", "No url")}\n' # Add URL of the paper
    doc_str += f'Year: {int(doc.metadata.get("year", " "))}\n' # Add year of the paper
    doc_str += f'Publisher: {doc.metadata.get("publisher", " ")}\n' # Add publisher of the paper
    doc_str += f'{doc.page_content}\n\n'
  return doc_str


def load_config(path: str = "config.yaml") -> dict:
    """
    Loads a YAML configuration file into a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
