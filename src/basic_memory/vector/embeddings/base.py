"""Base embedding provider implementation."""

from abc import ABC, abstractmethod
from typing import List

from basic_memory.vector.base import EmbeddingProvider


class BaseEmbeddingProvider(EmbeddingProvider):
    """Base implementation of embedding provider with common functionality."""

    def __init__(self):
        self._initialized = False
        self._dimension = None

    async def initialize(self) -> None:
        """Initialize the embedding provider."""
        if self._initialized:
            return
        await self._initialize_impl()
        self._initialized = True

    async def close(self) -> None:
        """Clean up resources."""
        if not self._initialized:
            return
        await self._close_impl()
        self._initialized = False

    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        if self._dimension is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        return self._dimension

    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Provider-specific initialization logic."""
        pass

    @abstractmethod
    async def _close_impl(self) -> None:
        """Provider-specific cleanup logic."""
        pass

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Default implementation that calls embed_text for each text."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        
        results = []
        for text in texts:
            embedding = await self.embed_text(text)
            results.append(embedding)
        return results
