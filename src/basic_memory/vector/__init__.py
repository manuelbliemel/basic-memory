"""Vector search package for basic-memory.

This package provides vector search capabilities to complement the existing
FTS5 fuzzy search functionality. It includes:

- Abstract provider interfaces for vector databases
- Concrete implementations (ChromaDB)
- Embedding providers (local and API-based)
- Text chunking utilities
- Result combination logic
"""

from .config import SearchStrategy, EmbeddingModelType, VectorSearchConfig, SearchConfig

__all__ = [
    "SearchStrategy",
    "EmbeddingModelType", 
    "VectorSearchConfig",
    "SearchConfig",
]
