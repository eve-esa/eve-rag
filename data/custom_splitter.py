from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
import re
from typing import List
# Make sure NLTK's punkt tokenizer is downloaded
import nltk
import logging


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class SentenceTextSplitter(TextSplitter):
    """
    A text splitter that splits text by sentences while preserving LaTeX and tables.
    """

    def __init__(self, chunk_size: int = 1800, chunk_overlap: int = 0):
        """
        Initialize the sentence splitter.

        Args:
            chunk_size: Maximum size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _find_latex_environments(self, text):
        """
        Identify all LaTeX environments in the text, handling nested environments correctly.

        Args:
            text: The text to analyze

        Returns:
            List of (start, end) tuples for all LaTeX environments
        """
        environments = []
        pos = 0

        while True:
            # Find the next \begin
            begin_pos = text.find("\\begin{", pos)
            if begin_pos == -1:
                break

            # Find the matching \end
            end_pos = self._find_matching_end(text, begin_pos)
            if end_pos == -1:
                # Skip this \begin if there's no matching \end
                pos = begin_pos + 6  # Move past "\begin"
                continue

            environments.append((begin_pos, end_pos))
            pos = end_pos

        return environments

    def _find_matching_end(self, text, begin_pos):
        """
        Find the matching \end{...} for a \begin{...} at the given position.
        Handles nested environments correctly.

        Args:
            text: The text to search in
            begin_pos: Position of the \begin{...} command

        Returns:
            Position of the end of the matching \end{...} command or -1 if not found
        """
        # Extract the environment name
        begin_match = re.search(r'\\begin\{([^}]+)\}', text[begin_pos:])
        if not begin_match:
            return -1

        env_name = begin_match.group(1)
        env_begin = f"\\begin{{{env_name}}}"
        env_end = f"\\end{{{env_name}}}"

        # Find the end of the current \begin command
        current_pos = begin_pos + len(env_begin)
        nesting_level = 1

        while nesting_level > 0 and current_pos < len(text):
            # Look for the next \begin or \end of the same environment
            begin_idx = text.find(env_begin, current_pos)
            end_idx = text.find(env_end, current_pos)

            # If no more begin/end tags, environment is not properly closed
            if end_idx == -1:
                return -1

            # If we find an end tag first or no more begin tags
            if begin_idx == -1 or end_idx < begin_idx:
                nesting_level -= 1
                current_pos = end_idx + len(env_end)
            else:
                nesting_level += 1
                current_pos = begin_idx + len(env_begin)

        return current_pos if nesting_level == 0 else -1

    def _identify_preserved_spans(self, text):
        """
        Identify all spans in the text that should be preserved atomically.

        Args:
            text: The text to analyze

        Returns:
            List of (start, end) tuples for preserved spans
        """
        preserved_spans = []

        # Find LaTeX environments (tables, equations, etc.)
        preserved_spans.extend(self._find_latex_environments(text))

        # Find inline and display math formulas
        # latex_formula_pattern = re.compile(r'\${1,2}[^$]+\${1,2}|\\\[[^\]]+\\\]|\\\([^)]+\\\)')
        # for match in latex_formula_pattern.finditer(text):
        #     is_inside_env = any(start <= match.start() and match.end() <= end
        #                         for start, end in preserved_spans)
        #     if not is_inside_env:
        #         preserved_spans.append((match.start(), match.end()))

        # Find markdown tables
        table_pattern = re.compile(r'(\|[^\n]+\|\n)((?:\|[^\n]+\|\n)+)')
        for match in table_pattern.finditer(text):
            is_inside_env = any(start <= match.start() and match.end() <= end
                                for start, end in preserved_spans)
            if not is_inside_env:
                preserved_spans.append((match.start(), match.end()))

        # Sort and merge overlapping spans
        if preserved_spans:
            preserved_spans.sort()
            merged_spans = []
            current_start, current_end = preserved_spans[0]

            for start, end in preserved_spans[1:]:
                if start <= current_end:  # Spans overlap
                    current_end = max(current_end, end)
                else:  # No overlap
                    merged_spans.append((current_start, current_end))
                    current_start, current_end = start, end

            merged_spans.append((current_start, current_end))
            preserved_spans = merged_spans

        return preserved_spans

    def tokenize_with_protection(self, text):
        # Store patterns that should be protected
        protected_patterns = []

        # Find and replace LaTeX formulas with placeholders
        def replace_protected(match):
            protected_patterns.append(match.group(0))
            return f"PROTECTED_PLACEHOLDER_{len(protected_patterns) - 1}"

        # Pattern to match LaTeX formulas enclosed in \[ \] or $ $
        latex_pattern = r'\\\[.*?\\\]|\$.*?\$'

        # Pattern to match figure references like "Fig. 2:" or "Table 1."
        figure_pattern = r'(Fig\.|Figure|Tab\.|Table|Eq\.|Equation)\s+\d+[\.:][^\.]*?'

        # Combine patterns
        combined_pattern = f"({latex_pattern})|({figure_pattern})"

        # Replace protected elements with placeholders
        protected_text = re.sub(combined_pattern, replace_protected, text, flags=re.DOTALL)

        # Tokenize the protected text
        sentences = nltk.sent_tokenize(protected_text)

        # Restore protected elements
        for i, sentence in enumerate(sentences):
            for j, protected in enumerate(protected_patterns):
                sentences[i] = sentences[i].replace(f"PROTECTED_PLACEHOLDER_{j}", protected)

        return sentences

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentences while preserving LaTeX content and tables.

        Args:
            text: The text to split

        Returns:
            List of text chunks respecting sentence boundaries and preserving special content
        """
        # First, identify spans that should be preserved
        preserved_spans = self._identify_preserved_spans(text)

        # Tokenize text into sentences, but skip the preserved spans
        sentences = []
        last_end = 0

        for start, end in preserved_spans:
            # Process text before the preserved span
            if start > last_end:
                before_text = text[last_end:start]
                if before_text.strip():
                    # Split the text before the preserved span into sentences
                    before_sentences = self.tokenize_with_protection(before_text)
                    sentences.extend(before_sentences)

            # Add the preserved span as a single "sentence"
            sentences.append(text[start:end])
            last_end = end

        # Process text after the last preserved span
        if last_end < len(text):
            after_text = text[last_end:]
            if after_text.strip():
                after_sentences = self.tokenize_with_protection(after_text)
                sentences.extend(after_sentences)

        # Now, group sentences into chunks of the specified size
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())

            # If sentence itself exceeds chunk size, keep it as a single chunk
            if sentence_size > self.chunk_size:
                # If we have accumulated sentences, add them as a chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Add the oversized sentence as its own chunk
                chunks.append(sentence)
                continue

            # If adding this sentence would exceed chunk size, start a new chunk
            if current_size + sentence_size + (1 if current_chunk else 0) > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + (1 if current_size > 0 else 0)  # +1 for space

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Add overlap if specified
        if self.chunk_overlap > 0:
            chunks_with_overlap = [chunks[0]]

            for i in range(1, len(chunks)):
                prev_chunk = chunks[i - 1]
                current_chunk = chunks[i]

                # Add the overlap from the previous chunk
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk.split()) >= self.chunk_overlap else prev_chunk
                chunks_with_overlap.append(overlap_text + current_chunk)

            return chunks_with_overlap

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks based on sentences.

        Args:
            documents: List of Documents to split

        Returns:
            List of Documents after splitting
        """
        splits = []
        for doc in documents:
            text_splits = self.split_text(doc.page_content)
            for text in text_splits:
                splits.append(Document(
                    page_content=text,
                    metadata=doc.metadata
                ))
        return splits
