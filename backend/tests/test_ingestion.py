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
