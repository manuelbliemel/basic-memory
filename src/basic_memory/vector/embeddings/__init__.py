"""Embedding providers for basic-memory vector search.

This package contains implementations for generating text embeddings using
different providers like local sentence-transformers models and OpenAI API.
"""

from .local_embedding import LocalEmbeddingProvider
from .openai_embedding import OpenAIEmbeddingProvider

__all__ = ["LocalEmbeddingProvider", "OpenAIEmbeddingProvider"]
