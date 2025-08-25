import json
import yaml

# Format retrieved documents:
def format_docs(docs):
    doc_str = ''
    for i, doc in enumerate(docs):
        doc_str += f'Score: {doc.metadata.get("score", "None")}\n'
        doc_str += f'Document n. {i+1}\n'
        doc_str += f'TITLE: {doc.metadata.get("title", "No title")}\n'

        header = doc.metadata.get("header", {})
        if isinstance(header, str):
            try:
                header = json.loads(header)
            except json.JSONDecodeError:
                header = {}
        if isinstance(header, dict):
            for key, value in header.items():
                doc_str += f'{key}: {value}\n'

        doc_str += f'URL: {doc.metadata.get("url", "No url")}\n'

        year = doc.metadata.get("year")
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = "Unknown"
        doc_str += f'Year: {year}\n'

        doc_str += f'Publisher: {doc.metadata.get("publisher", "Unknown")}\n'
        doc_str += f'{doc.page_content}\n\n'

    return doc_str


def load_config(path: str = "config.yaml") -> dict:
    """
    Loads a YAML configuration file into a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
