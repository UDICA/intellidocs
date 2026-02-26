from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around sentence-transformers for dense and sparse embeddings.

    Provides two embedding modes:
    - Dense: high-dimensional normalized vectors for semantic similarity search
    - Sparse: BM25-style token frequency vectors for keyword matching

    The sparse representation uses the model's own tokenizer to produce token IDs
    as indices and term frequencies as values, making it compatible with Qdrant's
    sparse vector format.

    Args:
        model_name: Name of the sentence-transformers model to load.
            Defaults to ``all-MiniLM-L6-v2`` (384-dim, fast, good quality).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model %s (dim=%d)", model_name, self.dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a list of texts.

        Returns L2-normalized vectors suitable for cosine similarity search.
        Normalization means cosine similarity reduces to a simple dot product,
        which is faster to compute in vector databases.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, each of length ``self.dimension``.
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def sparse_embed(self, texts: list[str]) -> list[dict]:
        """Generate sparse BM25-style embeddings using term frequencies.

        Uses the model's own wordpiece tokenizer to convert text into token IDs,
        then counts occurrences of each token. This produces a sparse vector
        where indices are token IDs and values are term frequencies.

        The output format is compatible with Qdrant sparse vectors, enabling
        hybrid search that combines dense semantic matching with sparse
        keyword matching.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of dicts, each containing:
                - ``indices``: sorted list of integer token IDs
                - ``values``: corresponding term frequency floats
        """
        results: list[dict] = []
        for text in texts:
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            indices = sorted(tf.keys())
            values = [float(tf[i]) for i in indices]
            results.append({"indices": indices, "values": values})
        return results

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text into integer token IDs using the model's tokenizer.

        Special tokens (CLS, SEP, PAD) are excluded so the sparse
        representation only contains content tokens.

        Args:
            text: Raw text string to tokenize.

        Returns:
            List of integer token IDs.
        """
        encoded = self.model.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return encoded["input_ids"]
