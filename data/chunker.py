import dataclasses
from typing import List
import logging
import time

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from .custom_splitter import SentenceTextSplitter

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

current_time = time.strftime("%Y_%m_%d_%H-%M-%S")

# Add file and stream handlers to the logger
#logger.addHandler(logging.FileHandler(f"logs/chunk_document_{current_time}.log"))
logger.addHandler(logging.StreamHandler())


@dataclasses.dataclass
class Section:
    """
    A class representing a section in a Markdown document.
    """
    level: int
    title: str
    header: str
    content: str
    subsections: List['Section'] = dataclasses.field(default_factory=list)


class MarkdownTwoStepChunker:
    """
    A class that implements a two-step chunking strategy for Markdown documents:
    1. Split the document into logical sections based on Markdown headers
    2. If any section exceeds the max chunk size, apply a secondary chunking method
       that preserves LaTeX formulas and tables
    3. Try to aggregate subsections into sections when possible
    4. Add word-based overlap between chunks
    """

    def __init__(self, max_chunk_size: int = 1800, chunk_overlap: int = 0, add_headers: bool = True,
                 word_overlap: int = None, headers_to_split_on: List[int] = None, merge_small_chunks: bool = False):
        """
        Initialize the chunker with configuration parameters.

        Args:
            max_chunk_size: Maximum size of any chunk in characters
            chunk_overlap: Number of characters to overlap between chunks in the secondary split
            add_headers: Whether to add headers to the chunks
            word_overlap: Number of words to overlap between chunks (defaults to None,
                         will estimate from chunk_overlap if not specified)
        """
        if headers_to_split_on is None:
            headers_to_split_on = [1, 2, 3, 4, 5, 6]
        self.headers_to_split_on = [("#" * level, level) for level in headers_to_split_on]
        self.max_chunk_size = max_chunk_size
        self.overlap = chunk_overlap
        self.word_overlap = word_overlap
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on, strip_headers=False
        )
        self.add_headers = add_headers
        # Use the custom splitter that preserves LaTeX and tables
        self.text_splitter = SentenceTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.merge_small_chunks = merge_small_chunks

    def _text_split(self, docs: list[Document]) -> List[Document]:
        """
        Split the text into chunks using the PreservingRecursiveCharacterTextSplitter.

        Args:
            docs: The documents to split

        Returns:
            List of chunks
        """
        return self.text_splitter.split_documents(docs)

    def _add_section_header(self, section: Document):
        """
        Add a header to the section content.

        Args:
            section: The section to process

        Returns:
            The section with added header
        """
        headers = section.metadata
        reverse_headers = headers.items()
        reverse_headers = sorted(reverse_headers, key=lambda x: x[0], reverse=True)
        for level, header in reverse_headers:
            header_str = f"{'#' * level} {header}"
            if header_str not in section.page_content:
                section.page_content = header_str + f"\n{section.page_content}"
        return section

    def _chunk_markdown(self, markdown_text: str) -> List[Document]:
        """
        Process the markdown text using the two-step chunking strategy.

        Args:
            markdown_text: The markdown document to process

        Returns:
            List of properly sized markdown chunks
        """
        logger.info(f"Markdown splitting started")
        sections = self.markdown_splitter.split_text(markdown_text)
        return sections

    def _merge_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Merge small chunks if they meet the criteria, with continuous merging.
        1. Their combined length doesn't exceed max_chunk_size
        2. They have compatible heading levels (same level or previous at higher level)

        Args:
            chunks: List of Document objects to consider for merging

        Returns:
            List of Documents after merging small chunks
        """
        if not chunks or len(chunks) <= 1:
            return chunks

        result = []
        i = 0

        while i < len(chunks):
            # Start with the current chunk
            current_chunk = chunks[i]
            i += 1

            # Try to merge with subsequent chunks as long as possible
            while i < len(chunks):
                next_chunk = chunks[i]

                # Get the lowest header level for each chunk (lower number = higher level)
                current_lowest_level = float('inf')
                next_lowest_level = float('inf')

                for level_str, _ in current_chunk.metadata.items():
                    try:
                        level = int(level_str)
                        current_lowest_level = min(current_lowest_level, level)
                    except ValueError:
                        continue

                for level_str, _ in next_chunk.metadata.items():
                    try:
                        level = int(level_str)
                        next_lowest_level = min(next_lowest_level, level)
                    except ValueError:
                        continue

                # Check if we can merge based on heading levels and combined length
                can_merge_headers = (
                        # No headers in either chunk
                        (current_lowest_level == float('inf') or next_lowest_level == float('inf')) or
                        # Same level headers
                        (current_lowest_level == next_lowest_level) or
                        # Previous chunk has higher level header (lower number)
                        (current_lowest_level < next_lowest_level)
                )

                combined_length = len(current_chunk.page_content.split()) + len(next_chunk.page_content.split())

                if can_merge_headers and combined_length <= self.max_chunk_size:
                    next_chunk_headers = ""
                    for level_str, header in next_chunk.metadata.items():
                        next_chunk_headers += f"{'#' * int(level_str)} {header}\n"
                    # Merge the chunks
                    current_chunk = Document(
                        page_content=current_chunk.page_content + "\n\n" + next_chunk_headers + "\n" + next_chunk.page_content,
                        metadata=current_chunk.metadata.copy()  # Keep the metadata of the first chunk
                    )
                    i += 1  # Move to the next chunk
                else:
                    # Cannot merge, stop this merging sequence
                    break

            # Add the final merged chunk (or original if no merging happened)
            result.append(current_chunk)

        return result

    def _add_word_overlap(self, chunks: List[str], overlap_words: int = None) -> List[str]:
        """
        Add overlap between adjacent chunks based on word count.

        Args:
            chunks: List of text chunks
            overlap_words: Number of words to overlap (calculates from self.overlap if None)

        Returns:
            List of chunks with word-based overlap added
        """
        if not chunks or len(chunks) <= 1:
            return chunks

        # If overlap_words is not specified, estimate it from character overlap
        if overlap_words is None:
            # Approximate average word length as 5 characters + 1 for space
            avg_word_length = 6
            overlap_words = max(1, self.overlap // avg_word_length)

        if overlap_words <= 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Split the previous chunk into words
            prev_words = prev_chunk.split()

            # Get the last overlap_words from the previous chunk
            overlap_text = ' '.join(prev_words[-overlap_words:]) if len(prev_words) >= overlap_words else prev_chunk

            # Add the overlap text to the beginning of the current chunk
            overlapped_chunk = overlap_text + " " + current_chunk

            result.append(overlapped_chunk)

        return result
    def set_max_chunk_size(self, max_chunk_size: int) -> None:
        """
        Set the maximum size of any chunk in characters.

        Args:
            max_chunk_size: Maximum size of any chunk in characters
        """
        self.max_chunk_size = max_chunk_size
        self.text_splitter.chunk_size = max_chunk_size
        logger.info(f"Max chunk size set to {max_chunk_size} characters")


    def chunk(self, markdown_text: str, overlap_words: int = None) -> List[str]:
        """
        Chunk the markdown text into sections and subsections.
        Merges small chunks when appropriate and adds word-based overlap between chunks.
        Preserves LaTeX formulas and tables as atomic units.

        Args:
            markdown_text: The markdown document to process
            overlap_words: Number of words to overlap between chunks (optional)

        Returns:
            List of sections and subsections with word-based overlap
        """
        logger.info(f"Markdown chunking started")
        sections = self._chunk_markdown(markdown_text)

        final_sections = []
        for section in sections:
            # Check if the section exceeds the max chunk size
            if len(section.page_content) > self.max_chunk_size:
                # If it does, split it further using the text splitter that preserves LaTeX and tables
                sub_sections = self._text_split([section])
                for sub_section in sub_sections:
                    final_sections.append(sub_section)
            else:
                # If it doesn't, just add it to the final sections
                final_sections.append(section)

        # Merge small chunks where possible
        if self.merge_small_chunks:
            final_sections = self._merge_small_chunks(final_sections)

        if self.add_headers:
            final_sections = [self._add_section_header(section) for section in final_sections]

        # Convert Document objects to strings
        final_sections = [section.page_content for section in final_sections]



        # Add word-based overlap between chunks
        #final_sections = self._add_word_overlap(final_sections)

        return final_sections
