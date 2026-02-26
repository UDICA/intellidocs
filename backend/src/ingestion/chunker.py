"""Recursive character text splitter with configurable chunk size and overlap.

Splits documents into manageable chunks for embedding, using a hierarchy of
separators (paragraph breaks -> line breaks -> sentences -> words -> characters)
to preserve semantic coherence within each chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk with metadata.

    Attributes:
        content: The text content of this chunk.
        metadata: Key-value pairs tracking provenance (source file,
            chunk index, etc.).
    """

    content: str
    metadata: dict[str, str | int] = field(default_factory=dict)


class TextChunker:
    """Recursive character text splitter with configurable overlap.

    Splits text by trying separators in priority order: paragraph breaks
    first, then line breaks, sentence endings, word boundaries, and finally
    character-level splits as a last resort. This preserves the most
    meaningful boundaries possible within the chunk size constraint.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters from the end of the previous
            chunk to prepend to the next chunk, providing context continuity.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        metadata: dict[str, str | int] | None = None,
    ) -> list[Chunk]:
        """Split text into overlapping chunks using recursive separators.

        Args:
            text: The input text to split.
            metadata: Optional base metadata to attach to every chunk.
                Each chunk also receives a ``chunk_index`` key automatically.

        Returns:
            A list of :class:`Chunk` objects with content and metadata.
        """
        base_metadata = metadata or {}
        raw_chunks = self._split_recursive(text, self.SEPARATORS)
        merged = self._merge_chunks(raw_chunks)

        return [
            Chunk(
                content=c,
                metadata={**base_metadata, "chunk_index": i},
            )
            for i, c in enumerate(merged)
        ]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that works.

        Tries each separator in order. When a separator produces pieces
        that fit within ``chunk_size``, those pieces are accumulated.
        Pieces that are still too large are split further using the
        remaining separators.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Hard split at chunk_size as last resort
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return self._split_recursive(text, [])

        parts = text.split(sep)
        results: list[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    results.append(current)
                if len(part) > self.chunk_size:
                    results.extend(
                        self._split_recursive(part, remaining_seps)
                    )
                    current = ""
                else:
                    current = part

        if current.strip():
            results.append(current)

        return results

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks for context continuity.

        Takes the tail of each chunk and prepends it to the next one,
        trying to break at a word boundary so the overlap reads naturally.
        """
        if len(chunks) <= 1:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = (
                prev[-self.chunk_overlap :]
                if len(prev) > self.chunk_overlap
                else prev
            )
            # Find a clean word boundary in the overlap
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1 :]
            merged = (
                overlap_text + " " + chunks[i] if overlap_text else chunks[i]
            )
            result.append(merged.strip())

        return result
