#!/usr/bin/env python3
"""
Q&A Dataset Generation Script

This script generates a Q&A evaluation dataset from documents using an LLM.
It validates that generated references exist in the Qdrant vector database.

Usage:
    python generate_qa_dataset.py --config config.yaml

Requirements:
    - YAML configuration file
    - Environment variables for API keys and Qdrant credentials
    - JSONL file with documents
    - Qdrant collection with document chunks
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import string
from collections import Counter

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from tqdm import tqdm


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

QA_GENERATION_PROMPT = """You are an agent that generates questions from a provided research paper. Your job is to generate one specific question and provide the relevant sections from the paper as references.

Instructions:

Generate a question that can be answered solely by the facts in the provided paper.

Extract up to 5 significant sections from the paper that answer the question. These must be *exact copies* from the text and should be whole sentences where possible.

Focus on the most relevant information; avoid background or unrelated sections.

Format the response in JSON with three fields:

"oath": "I will not use the word 'and' in the question unless it is part of a proper noun. I will also make sure the question is concise."

"question": A concise question directly answerable using the references.

"references": A list of the extracted sections from the paper.

Notes:

Make the question specific; do not ask about multiple topics.

DO NOT USE THE WORD 'and' IN THE QUESTION UNLESS IT IS PART OF A PROPER NOUN.

Do not repeat a question that has already been used.

When the paper is long, scan all sections but only pick the most relevant ones to answer the question.

Example:

Paper Text:
"Section 1: Introduction: Climate change has accelerated glacier melt in the Himalayas, affecting water resources downstream.

Section 2: Methodology: Remote sensing data from 2000–2020 were analyzed to quantify changes in glacier area.

Section 3: Results: Glacier area decreased by 12% over 20 years, with the highest retreat in the eastern Himalayas. Streamflow measurements confirmed increased seasonal variability.

Section 4: Discussion: The retreat impacts hydropower generation and agriculture. Communities relying on glacier-fed rivers experience water stress during summer months.

Section 5: Conclusion: Urgent adaptation strategies are needed to mitigate the socioeconomic impact of glacier retreat."


Example Output:
{
  "oath": "I will not use the word 'and' in the question unless it is part of a proper noun. I will also make sure the question is concise.",
  "question": "How has glacier retreat affected downstream water resources in the Himalayas?",
  "references": [
    "Section 3: Results: Glacier area decreased by 12% over 20 years, with the highest retreat in the eastern Himalayas. Streamflow measurements confirmed increased seasonal variability.",
    "Section 4: Discussion: The retreat impacts hydropower generation and agriculture. Communities relying on glacier-fed rivers experience water stress during summer months."
  ]
}

Please provide your answer in the following JSON format:
{format_instructions}"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def is_reference_present_fuzzy(reference: str, document: str, threshold: float = 0.8) -> bool:
    """Check if reference appears in document with fuzzy matching."""
    ref_tokens = normalize_text(reference).split()
    doc_tokens = normalize_text(document).split()
    if not ref_tokens:
        return False
    matched_tokens = sum(1 for t in ref_tokens if t in doc_tokens)
    fraction_matched = matched_tokens / len(ref_tokens)
    return fraction_matched >= threshold


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """Client for interacting with LLM API."""

    def __init__(self, config: Dict):
        """Initialize LLM client."""
        load_dotenv()

        # Support both direct credentials and environment variables
        if 'api_key' in config['model']:
            # Direct credential in config
            api_key = config['model']['api_key']
            base_url = config['model'].get('base_url', 'https://api.openai.com/v1')
        else:
            # Environment variable
            api_key = os.getenv(config['model']['api_key_env'])
            base_url = os.getenv(
                config['model'].get('base_url_env', ''),
                config['model'].get('base_url_default', 'https://api.openai.com/v1')
            )

        if not api_key:
            raise ValueError(
                "API key not found! Either:\n"
                "1. Set 'api_key' directly in config.yaml, OR\n"
                f"2. Set {config['model'].get('api_key_env', 'OPENAI_API_KEY')} environment variable"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = config['model']['name']
        self.max_tokens = config['model']['max_tokens']
        self.temperature = config['model']['temperature']

        logging.info(f"✓ LLM client initialized: {self.model_name}")

    def generate(self, prompt: str) -> str:
        """Generate text using the LLM."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content


# ============================================================================
# QDRANT CLIENT
# ============================================================================

class QdrantManager:
    """Manager for Qdrant operations."""

    def __init__(self, config: Dict):
        """Initialize Qdrant client."""
        load_dotenv()

        # Support both direct credentials and environment variables
        if 'url' in config['qdrant'] and 'api_key' in config['qdrant']:
            # Direct credentials in config
            url = config['qdrant']['url']
            api_key = config['qdrant']['api_key']
        else:
            # Environment variables
            url = os.getenv(config['qdrant']['url_env'])
            api_key = os.getenv(config['qdrant']['api_key_env'])

        if not url or not api_key:
            raise ValueError(
                "Qdrant credentials not found! Either:\n"
                "1. Set 'url' and 'api_key' directly in config.yaml, OR\n"
                f"2. Set {config['qdrant'].get('url_env', 'QDRANT_URL')} and "
                f"{config['qdrant'].get('api_key_env', 'QDRANT_API_KEY')} environment variables"
            )

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = config['qdrant']['collection_name']
        self.file_path_field = config['qdrant']['file_path_field']
        self.content_field = config['qdrant']['content_field']

        # Verify connection
        info = self.client.get_collection(self.collection_name)
        logging.info(f"✓ Connected to Qdrant collection: {self.collection_name}")
        logging.info(f"  Points: {info.points_count}, Status: {info.status}")

    def get_document_chunks(self, file_path: str) -> List[Tuple[int, str]]:
        """
        Retrieve all chunks for a specific document from Qdrant.

        Returns:
            List of (chunk_number, content) tuples
        """
        chunks = []
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=self.file_path_field,
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for record in records:
                content = record.payload.get(self.content_field, '') or record.payload.get('text', '')
                chunks.append(content)

            if offset is None:
                break

        # Number chunks sequentially (0-indexed)
        numbered_chunks = [(i, chunk) for i, chunk in enumerate(chunks)]
        return numbered_chunks


# ============================================================================
# Q&A GENERATOR
# ============================================================================

class QAGenerator:
    """Generate Q&A pairs from documents."""

    def __init__(self, config: Dict, llm_client: LLMClient, qdrant_manager: QdrantManager):
        """Initialize Q&A generator."""
        self.config = config
        self.llm = llm_client
        self.qdrant = qdrant_manager
        self.threshold = config['validation']['fuzzy_threshold']

        # Statistics
        self.stats = {
            'documents_processed': 0,
            'qa_pairs_generated': 0,
            'qa_pairs_skipped': 0,
            'total_refs_generated': 0,
            'total_refs_valid': 0,
            'total_refs_removed': 0
        }

    def generate_qa_from_document(self, doc_content: str, file_path: str) -> Optional[Dict]:
        """
        Generate a Q&A pair from a document.

        Returns:
            Q&A pair dict or None if generation fails or no valid references
        """
        # Get all chunks for this document from Qdrant
        numbered_chunks = self.qdrant.get_document_chunks(file_path)

        if len(numbered_chunks) == 0:
            logging.warning(f"No chunks found in Qdrant for: {file_path}")
            self.stats['qa_pairs_skipped'] += 1
            return None

        # Generate Q&A using LLM
        prompt = f"Context:\\n{doc_content}\\n\\nInstructions:\\n{QA_GENERATION_PROMPT}"
        prompt = prompt.replace("{format_instructions}",
                               '{"oath": "...", "question": "...", "references": [...]}')

        try:
            response = self.llm.generate(prompt)

            # Parse JSON response
            # Handle cases where LLM wraps response in markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            qa_output = json.loads(response)

            if 'question' not in qa_output or 'references' not in qa_output:
                logging.warning(f"Invalid LLM response format for {file_path}")
                self.stats['qa_pairs_skipped'] += 1
                return None

        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to parse LLM response for {file_path}: {e}")
            self.stats['qa_pairs_skipped'] += 1
            return None

        # Validate references
        validated_references = []
        refs_generated = len(qa_output['references'])
        self.stats['total_refs_generated'] += refs_generated

        for ref_text in qa_output['references']:
            # Check if this reference appears in at least one chunk
            found_in_chunks = []
            for chunk_num, chunk_content in numbered_chunks:
                if is_reference_present_fuzzy(ref_text, chunk_content, threshold=self.threshold):
                    found_in_chunks.append(chunk_num)

            if found_in_chunks:
                validated_references.append({
                    "text": ref_text,
                    "chunk_numbers": found_in_chunks
                })
                self.stats['total_refs_valid'] += 1
            else:
                logging.debug(f"Reference not found in chunks: '{ref_text[:60]}...'")
                self.stats['total_refs_removed'] += 1

        # Only keep Q&A pair if at least one valid reference remains
        if len(validated_references) > 0:
            qa_entry = {
                "question": qa_output['question'],
                "file_path": file_path,
                "references": validated_references,
                "source_document": doc_content[:500] + "..." if len(doc_content) > 500 else doc_content,
                "refs_generated": refs_generated,
                "refs_valid": len(validated_references),
                "refs_removed": refs_generated - len(validated_references)
            }

            self.stats['qa_pairs_generated'] += 1
            logging.info(f"✓ Generated Q&A with {len(validated_references)}/{refs_generated} valid references")
            return qa_entry
        else:
            logging.warning(f"No valid references found for Q&A from {file_path}")
            self.stats['qa_pairs_skipped'] += 1
            return None

    def print_statistics(self):
        """Print generation statistics."""
        logging.info("\n" + "="*60)
        logging.info("Q&A GENERATION STATISTICS")
        logging.info("="*60)
        logging.info(f"Documents processed: {self.stats['documents_processed']}")
        logging.info(f"Q&A pairs generated: {self.stats['qa_pairs_generated']}")
        logging.info(f"Q&A pairs skipped: {self.stats['qa_pairs_skipped']}")

        if self.stats['documents_processed'] > 0:
            success_rate = (self.stats['qa_pairs_generated'] / self.stats['documents_processed']) * 100
            logging.info(f"Success rate: {success_rate:.1f}%")

        logging.info(f"\nReference Statistics:")
        logging.info(f"  Total references generated: {self.stats['total_refs_generated']}")
        logging.info(f"  Valid references: {self.stats['total_refs_valid']}")
        logging.info(f"  Removed references: {self.stats['total_refs_removed']}")

        if self.stats['total_refs_generated'] > 0:
            validity_rate = (self.stats['total_refs_valid'] / self.stats['total_refs_generated']) * 100
            logging.info(f"  Validity rate: {validity_rate:.1f}%")

        logging.info("="*60)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def load_documents(jsonl_path: str) -> List[Tuple[str, str]]:
    """
    Load documents from JSONL file.

    Returns:
        List of (content, file_path) tuples
    """
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            content = doc.get('content', '')
            file_path = doc.get('metadata', {}).get('file_path', '')
            if content and file_path:
                documents.append((content, file_path))

    logging.info(f"✓ Loaded {len(documents)} documents from {jsonl_path}")
    return documents


def sample_documents(documents: List[Tuple[str, str]], n_samples: int, random_seed: Optional[int] = None) -> List[Tuple[str, str]]:
    """Sample documents randomly."""
    if random_seed is not None:
        random.seed(random_seed)

    sampled = random.sample(documents, min(n_samples, len(documents)))
    logging.info(f"✓ Sampled {len(sampled)} documents")
    return sampled


def save_dataset(dataset: List[Dict], output_file: str, overwrite: bool = False):
    """Save Q&A dataset to JSON file."""
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # Check if file exists
    if os.path.exists(output_file) and not overwrite:
        logging.error(f"Output file already exists: {output_file}")
        logging.error("Set 'overwrite: true' in config to overwrite")
        return False

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logging.info(f"✓ Dataset saved to: {output_file}")
    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate Q&A evaluation dataset from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python generate_qa_dataset.py --config config.yaml

Make sure to set environment variables:
    - OPENAI_API_KEY (or your configured API key env var)
    - QDRANT_URL
    - QDRANT_API_KEY
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        level=config['logging']['level'],
        log_file=config['logging'].get('log_file')
    )

    logging.info("="*60)
    logging.info("Q&A DATASET GENERATION")
    logging.info("="*60)
    logging.info(f"Configuration: {args.config}")

    # Initialize clients
    llm_client = LLMClient(config)
    qdrant_manager = QdrantManager(config)

    # Initialize generator
    qa_generator = QAGenerator(config, llm_client, qdrant_manager)

    # Load and sample documents
    logging.info(f"\nLoading documents from: {config['input']['documents_jsonl']}")
    documents = load_documents(config['input']['documents_jsonl'])

    sampled_docs = sample_documents(
        documents,
        config['input']['num_samples'],
        config['input'].get('random_seed')
    )

    # Generate Q&A pairs
    logging.info(f"\nGenerating Q&A pairs...")
    logging.info("This may take several minutes depending on the number of documents and model speed.\n")

    qa_dataset = []
    progress_file = config['output'].get('progress_file') if config['output'].get('save_progress') else None

    for idx, (doc_content, file_path) in enumerate(tqdm(sampled_docs, desc="Processing documents")):
        qa_generator.stats['documents_processed'] += 1

        qa_pair = qa_generator.generate_qa_from_document(doc_content, file_path)

        if qa_pair is not None:
            qa_dataset.append(qa_pair)

        # Save progress periodically
        if progress_file and (idx + 1) % 5 == 0:
            save_dataset(qa_dataset, progress_file, overwrite=True)

    # Print statistics
    qa_generator.print_statistics()

    # Save final dataset
    if len(qa_dataset) > 0:
        success = save_dataset(
            qa_dataset,
            config['output']['output_file'],
            overwrite=config['output'].get('overwrite', False)
        )

        if success:
            logging.info(f"\n✓ Successfully generated {len(qa_dataset)} Q&A pairs!")
            logging.info(f"  Output: {config['output']['output_file']}")
        else:
            logging.error("\n✗ Failed to save dataset")
            return 1
    else:
        logging.error("\n✗ No Q&A pairs generated")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
