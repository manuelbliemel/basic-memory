"""Configuration classes and enums for vector search."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SearchStrategy(str, Enum):
    """Search execution strategy options."""
    FUZZY_ONLY = "fuzzy_only"           # Use only FTS5 search
    VECTOR_ONLY = "vector_only"         # Use only vector search  
    FUZZY_PRIMARY = "fuzzy_primary"     # Fuzzy first, fallback to vector
    VECTOR_PRIMARY = "vector_primary"   # Vector first, fallback to fuzzy
    HYBRID = "hybrid"                   # Both searches, union results


class EmbeddingModelType(str, Enum):
    """Embedding model provider types."""
    LOCAL = "local"      # sentence-transformers models
    OPENAI = "openai"    # OpenAI API embeddings
    CUSTOM = "custom"    # Custom provider implementation


class VectorSearchConfig(BaseModel):
    """Vector search configuration."""
    enabled: bool = True
    provider: str = "chromadb"
    embedding_model: EmbeddingModelType = EmbeddingModelType.LOCAL
    embedding_model_name: str = "all-MiniLM-L6-v2"
    api_key: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50
    similarity_threshold: float = 0.1
    max_results: int = 5


class SearchConfig(BaseModel):
    """Combined search configuration."""
    strategy: SearchStrategy = SearchStrategy.HYBRID
    vector: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
