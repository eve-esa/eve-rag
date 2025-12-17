from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
import json
import argparse
from tqdm import tqdm
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """
    Loads a YAML configuration file into a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def update_payload_by_filepath(client, collection_name, file_path, payload_updates, dry_run=False):
    """
    Update payload fields for points matching a specific file_path.
    Only updates fields that are present in payload_updates, preserving other existing fields.

    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        file_path: The file_path value to match in existing points
        payload_updates: Dictionary of fields to update (fields not included will be preserved)
        dry_run: If True, only show what would be updated without making changes

    Returns:
        Number of points updated (or would be updated in dry-run mode)
    """
    try:
        # Search for points with matching file_path
        points = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="file_path",
                        match=MatchValue(value=file_path)
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=10000
        )

        point_ids = [point.id for point in points[0]]

        if not point_ids:
            if not dry_run:
                print(f"Warning: No points found with file_path='{file_path}'")
            return 0

        if dry_run:
            print(f"[DRY RUN] Would update {len(point_ids)} points with file_path='{file_path}'")
            print(f"[DRY RUN] New payload: {payload_updates}")
        else:
            # Update payload for all matching points
            # set_payload merges with existing payload by default (doesn't overwrite unlisted fields)
            client.set_payload(
                collection_name=collection_name,
                payload=payload_updates,
                points=point_ids
            )

        return len(point_ids)

    except Exception as e:
        print(f"Error updating payload for file_path '{file_path}': {e}")
        return 0


def process_updates_from_mapping(client, collection_name, mapping_path, jsonl_path=None, dry_run=False):
    """
    Process file_path updates from a mapping.json file and optionally additional metadata from a JSONL file.

    The mapping.json should contain key-value pairs where:
    - key: new file_path to update to
    - value: old file_path to search for in Qdrant

    The JSONL file (optional) should contain additional metadata to update, with each line having:
    - file_path: the new file path (key from mapping.json) to match metadata
    - Other fields: additional metadata to update

    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        mapping_path: Path to mapping.json file
        jsonl_path: Path to JSONL file with additional metadata (optional)
        dry_run: If True, only show what would be updated without making changes
    """
    total_updated = 0
    total_processed = 0

    # Load the mapping file
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    # Load additional metadata from JSONL if provided
    metadata_dict = {}
    if jsonl_path:
        print(f"Loading additional metadata from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'file_path' in entry:
                        file_path = entry['file_path']
                        # Store all fields except file_path as metadata
                        metadata = {k: v for k, v in entry.items() if k != 'file_path'}
                        metadata_dict[file_path] = metadata
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
        print(f"Loaded metadata for {len(metadata_dict)} files")

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN MODE - No changes will be made to the database")
        print(f"{'='*60}\n")

    print(f"Processing {len(mapping)} file path mappings...")

    for new_path, old_path in tqdm(mapping.items(), desc="Updating file paths"):
        try:
            # Start with file_path update
            payload_updates = {"file_path": new_path}

            # Add additional metadata if available for this new_path
            if jsonl_path and old_path in metadata_dict:
                additional_metadata = metadata_dict[old_path]
                # Clean and prepare additional metadata
                for key, value in additional_metadata.items():
                    # Handle year conversion
                    if key == 'year':
                        if value is not None:
                            try:
                                payload_updates[key] = int(float(value))
                            except:
                                payload_updates[key] = None
                        else:
                            payload_updates[key] = None
                    # Handle n_citations conversion
                    elif key == 'n_citations':
                        if value is not None:
                            try:
                                payload_updates[key] = int(float(value))
                            except:
                                payload_updates[key] = None
                        else:
                            payload_updates[key] = None
                    # String fields
                    elif key in ['title', 'journal', 'content']:
                        if value is not None:
                            payload_updates[key] = str(value)
                        else:
                            payload_updates[key] = None
                    elif key == 'file_path':
                        # Skip file_path as it's already handled
                        continue
                    else:
                        # For any other field, use the value as-is
                        payload_updates[key] = value

            # Update payload (searches by new_path)
            num_updated = update_payload_by_filepath(
                client,
                collection_name,
                new_path,
                payload_updates,
                dry_run=dry_run
            )

            total_updated += num_updated
            total_processed += 1

        except Exception as e:
            print(f"Error processing mapping '{old_path}' -> '{new_path}': {e}")
            continue

    print(f"\nUpdate Summary:")
    print(f"  Mappings processed: {total_processed}")
    if dry_run:
        print(f"  Total points that would be updated: {total_updated}")
        print(f"\n{'='*60}")
        print(f"DRY RUN COMPLETE - No changes were made")
        print(f"{'='*60}")
    else:
        print(f"  Total points updated: {total_updated}")
    return total_updated


def main():
    parser = argparse.ArgumentParser(
        description="Update file_path fields in Qdrant collection using a mapping.json file. "
                    "Searches for points using old paths (values) and updates them to new paths (keys)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file containing qdrant_url, qdrant_api_key, collection_name, mapping_path, and optionally jsonl_path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode to see what changes would be made without actually updating the database"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get Qdrant connection details from config
    qdrant_url = config.get("qdrant_url")
    qdrant_api_key = config.get("qdrant_api_key")
    collection_name = config.get("collection_name")
    mapping_path = config.get("mapping_path")
    jsonl_path = config.get("jsonl_path")  # Optional

    if not qdrant_url:
        raise ValueError("'qdrant_url' not found in config file")
    if not qdrant_api_key:
        raise ValueError("'qdrant_api_key' not found in config file")
    if not collection_name:
        raise ValueError("'collection_name' not found in config file")
    if not mapping_path:
        raise ValueError("'mapping_path' not found in config file")

    # Verify mapping file exists
    mapping_path = os.path.abspath(mapping_path)
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"Mapping file does not exist: {mapping_path}")

    # Verify JSONL file exists if provided
    if jsonl_path:
        jsonl_path = os.path.abspath(jsonl_path)
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")

    # Connect to Qdrant
    print(f"Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Verify collection exists
    if not client.collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist")

    print(f"Updating collection: {collection_name}")

    # Process updates
    process_updates_from_mapping(client, collection_name, mapping_path, jsonl_path=jsonl_path, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nPayload updates completed!")
    else:
        print("\nDry run completed!")


if __name__ == "__main__":
    main()