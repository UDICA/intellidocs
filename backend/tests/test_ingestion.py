from pathlib import Path

import pytest


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("This is a test document.\nIt has two lines.")
    return f


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    f = tmp_path / "sample.md"
    f.write_text("# Title\n\nSome markdown content.\n\n## Section\n\nMore text here.")
    return f


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    f = tmp_path / "sample.csv"
    f.write_text("name,description\nAlice,Engineer\nBob,Designer")
    return f


class TestDocumentLoader:
    def test_load_txt(self, sample_txt: Path):
        from backend.src.ingestion.loader import DocumentLoader

        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        assert len(docs) >= 1
        assert "test document" in docs[0].content
        assert docs[0].metadata["source"] == str(sample_txt)
        assert docs[0].metadata["format"] == "txt"

    def test_load_markdown(self, sample_md: Path):
        from backend.src.ingestion.loader import DocumentLoader

        loader = DocumentLoader()
        docs = loader.load(sample_md)
        assert len(docs) >= 1
        assert "markdown content" in docs[0].content

    def test_load_csv(self, sample_csv: Path):
        from backend.src.ingestion.loader import DocumentLoader

        loader = DocumentLoader()
        docs = loader.load(sample_csv)
        assert len(docs) >= 1
        assert "Alice" in docs[0].content

    def test_load_unsupported_format(self, tmp_path: Path):
        from backend.src.ingestion.loader import DocumentLoader

        f = tmp_path / "file.xyz"
        f.write_text("content")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(f)

    def test_load_directory(self, tmp_path: Path):
        from backend.src.ingestion.loader import DocumentLoader

        (tmp_path / "a.txt").write_text("Doc A")
        (tmp_path / "b.txt").write_text("Doc B")
        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)
        assert len(docs) == 2


class TestTextChunker:
    def test_chunk_short_text(self):
        from backend.src.ingestion.chunker import TextChunker

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("Short text.")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_chunk_long_text_produces_overlapping_chunks(self):
        from backend.src.ingestion.chunker import TextChunker

        text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50
        chunker = TextChunker(chunk_size=60, chunk_overlap=10)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # Each chunk should be within size limit (with some tolerance for overlap merging)
        for chunk in chunks:
            assert len(chunk.content) <= 80  # allow some tolerance

    def test_chunk_preserves_metadata(self):
        from backend.src.ingestion.chunker import TextChunker

        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk(
            "Word " * 20,
            metadata={"source": "test.txt"},
        )
        assert all(c.metadata["source"] == "test.txt" for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_splits_on_separators(self):
        from backend.src.ingestion.chunker import TextChunker

        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk(text)
        # Should prefer splitting on double newlines
        assert any("Paragraph one." in c.content for c in chunks)
