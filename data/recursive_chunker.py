import re
from typing import List
from langchain_text_splitters.base import TextSplitter

def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs, preserving all formatting."""
    return [p for p in re.split(r'\n\s*\n', text) if p.strip()]

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving original whitespace and punctuation."""
    # Match punctuation followed by whitespace/newline
    pattern = r'(?<=[.!?])(\s+)'
    parts = re.split(pattern, text)
    sentences = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            sentence += parts[i + 1]  # keep original whitespace
        sentences.append(sentence)
    return [s for s in sentences if s.strip()]

class RecursiveMarkdownSplitter(TextSplitter):
    """
    Recursive chunker that preserves Markdown and formatting.
    Splits at paragraphs -> sentences -> words without modifying original text.
    """


    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 0):
        super().__init__(keep_separator=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        return self._recursive_split(text)
    
    def chunk(self, text: str) -> List[str]:
        return self._recursive_split(text)

    def _count_words(self, text: str) -> int:
        return len(text.split())

    def _recursive_split(self, text: str) -> List[str]:
        if self._count_words(text) <= self.chunk_size:
            return [text]

        # 1. splitting by paragraphs
        paragraphs = _split_paragraphs(text)
        if len(paragraphs) > 1:
            chunks = []
            current = ""
            for p in paragraphs:
                if self._count_words(current + "\n\n" + p) <= self.chunk_size or not current:
                    current += ("\n\n" if current else "") + p
                else:
                    chunks.extend(self._recursive_split(current))
                    current = p
            if current:
                chunks.extend(self._recursive_split(current))
            return chunks

        # 2. splitting by sentences
        sentences = _split_sentences(text)
        if len(sentences) > 1:
            chunks = []
            current = ""
            for s in sentences:
                if self._count_words(current + s) <= self.chunk_size or not current:
                    current += s
                else:
                    chunks.append(current)
                    current = s
            if current:
                chunks.append(current)
            return chunks

        # 3. Fallback: split by words, but slice original text
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(chunk_text)
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
        return chunks
