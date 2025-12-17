from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
import yaml
import argparse

def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_file_path_index(config_path: str):
    """Create a keyword index for the file_path field in Qdrant."""
    config = load_config(config_path)

    QDRANT_URL = config.get("qdrant_url")
    QDRANT_API_KEY = config.get("qdrant_api_key")
    collection_name = config.get("collection_name")

    if not QDRANT_URL or not QDRANT_API_KEY or not collection_name:
        raise ValueError("Missing required config values: qdrant_url, qdrant_api_key, or collection_name")

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Verify collection exists
    if not client.collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist")

    print(f"Creating keyword index for 'file_path' in collection '{collection_name}'...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="file_path",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    print("Index created successfully!")
    print("\nYou can now run your update_qdrant_payload.py script.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a keyword index for the file_path field in Qdrant collection"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., update_config.yaml)"
    )
    args = parser.parse_args()

    create_file_path_index(args.config)