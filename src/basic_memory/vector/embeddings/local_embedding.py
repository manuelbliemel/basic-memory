"""Local embedding provider using sentence-transformers."""

import asyncio
from typing import List

from loguru import logger

from .base import BaseEmbeddingProvider


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using local sentence-transformers models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.model = None

    async def _initialize_impl(self) -> None:
        """Initialize the sentence-transformer model."""
        try:
            # Import here to avoid dependency issues if not installed
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers package is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Loading sentence-transformer model: {self.model_name}")
        
        # Load model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None, 
            SentenceTransformer, 
            self.model_name
        )
        
        # Get embedding dimension
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self._dimension}")

    async def _close_impl(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            # sentence-transformers doesn't have explicit cleanup
            self.model = None
            logger.debug("Local embedding model resources cleaned up")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text."""
        if not self._initialized or self.model is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._dimension

        # Run embedding generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            self.model.encode,
            text
        )
        
        return embedding.tolist()

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently using batch processing."""
        if not self._initialized or self.model is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        if not texts:
            return []

        # Handle empty texts
        processed_texts = []
        empty_indices = []
        for i, text in enumerate(texts):
            if text.strip():
                processed_texts.append(text)
            else:
                empty_indices.append(i)

        if not processed_texts:
            # All texts are empty
            return [[0.0] * self._dimension for _ in texts]

        # Run batch embedding generation in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.model.encode,
            processed_texts
        )

        # Convert to list and handle empty texts
        results = []
        processed_idx = 0
        
        for i in range(len(texts)):
            if i in empty_indices:
                results.append([0.0] * self._dimension)
            else:
                results.append(embeddings[processed_idx].tolist())
                processed_idx += 1

        return results
