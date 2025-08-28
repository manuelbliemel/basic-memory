"""Abstract base classes for vector search providers and embedding providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class VectorSearchProvider(ABC):
    """Abstract interface for vector database providers.
    
    This interface allows switching between different vector database
    implementations like ChromaDB, Pinecone, Weaviate, etc.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Set up vector database connection and initialize collections/indices."""
        pass

    @abstractmethod
    async def index_document(
        self, 
        id: str, 
        content: str, 
        metadata: Dict[str, Any],
        embeddings: Optional[List[float]] = None
    ) -> None:
        """Add or update a document in the vector index.
        
        Args:
            id: Unique identifier for the document
            content: Text content to be indexed
            metadata: Additional metadata to store with the document
            embeddings: Pre-computed embeddings (if None, will be computed by provider)
        """
        pass

    @abstractmethod
    async def search(
        self, 
        query_embeddings: List[float], 
        limit: int = 10,
        threshold: float = 0.3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute similarity search using query embeddings.
        
        Args:
            query_embeddings: Vector representation of the search query
            limit: Maximum number of results to return
            threshold: Minimum similarity score for results
            metadata_filter: Optional filter criteria for metadata
            
        Returns:
            List of matching documents with scores and metadata
        """
        pass

    @abstractmethod
    async def delete_document(self, id: str) -> bool:
        """Remove a document from the vector index.
        
        Args:
            id: Unique identifier of the document to remove
            
        Returns:
            True if document was deleted, False if document didn't exist
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections and resources."""
        pass


class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation.
    
    This interface allows using different embedding models and providers,
    from local sentence-transformers to OpenAI API to custom implementations.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text.
        
        Args:
            text: Input text to generate embeddings for
            
        Returns:
            List of float values representing the text embedding
        """
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts to generate embeddings for
            
        Returns:
            List of embedding vectors, one for each input text
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.
        
        Returns:
            Integer representing the embedding vector dimension
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding provider (load models, setup connections, etc)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (unload models, close connections, etc)."""
        pass
