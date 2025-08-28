"""ChromaDB vector search provider implementation."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger

from basic_memory.vector.base import VectorSearchProvider, EmbeddingProvider


class ChromaDBProvider(VectorSearchProvider):
    """ChromaDB implementation of vector search provider."""

    def __init__(
        self, 
        embedding_provider: EmbeddingProvider,
        persist_directory: Optional[Path] = None,
        collection_name: str = "basic_memory"
    ):
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory or Path.home() / ".basic-memory" / "chroma"
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Import here to avoid dependency issues if not installed
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise RuntimeError(
                "chromadb package is required for ChromaDB provider. "
                "Install it with: pip install chromadb"
            )

        logger.info(f"Initializing ChromaDB provider with persist directory: {self.persist_directory}")
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )

        # Initialize embedding provider
        await self.embedding_provider.initialize()
        
        # Get or create collection
        embedding_dimension = self.embedding_provider.get_dimension()
        
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Using existing ChromaDB collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"embedding_dimension": embedding_dimension}
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")

        logger.info("ChromaDB provider initialized successfully")

    async def index_document(
        self, 
        id: str, 
        content: str, 
        metadata: Dict[str, Any],
        embeddings: Optional[List[float]] = None
    ) -> None:
        """Add or update a document in the ChromaDB collection."""
        if self.collection is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = await self.embedding_provider.embed_text(content)

        # ChromaDB requires string values for metadata
        processed_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                processed_metadata[key] = str(value)

        # Run ChromaDB operations in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._upsert_document,
            id,
            content,
            embeddings,
            processed_metadata
        )

        logger.debug(f"Indexed document with ID: {id}")

    def _upsert_document(
        self,
        id: str,
        content: str, 
        embeddings: List[float],
        metadata: Dict[str, str]
    ) -> None:
        """Synchronous document upsert for thread pool execution."""
        self.collection.upsert(
            ids=[id],
            documents=[content],
            embeddings=[embeddings],
            metadatas=[metadata]
        )

    async def search(
        self, 
        query_embeddings: List[float], 
        limit: int = 10,
        threshold: float = 0.3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute similarity search using query embeddings."""
        if self.collection is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        # Process metadata filter for ChromaDB format
        where_clause = None
        if metadata_filter:
            where_clause = {}
            for key, value in metadata_filter.items():
                if value is not None:
                    where_clause[key] = {"$eq": str(value)}

        # Run search in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._execute_search,
            query_embeddings,
            limit,
            where_clause
        )

        # Filter by threshold and format results
        filtered_results = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns cosine distance - normalize to 0-1 similarity for consistency
            distance = results["distances"][0][i] if results["distances"][0] else 0.0
            
            # Use robust distance-to-similarity conversion
            # For cosine distance: similarity = max(0, 1 - distance/2) handles both normalized and non-normalized vectors
            similarity = max(0.0, 1.0 - distance / 2.0)
            
            if similarity >= threshold:
                result = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"][0] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                    "score": similarity,
                    "score_type": "vector_similarity"  # Mark score type for proper handling
                }
                filtered_results.append(result)

        logger.debug(f"Vector search returned {len(filtered_results)} results above threshold {threshold}")
        return filtered_results

    def _execute_search(
        self,
        query_embeddings: List[float],
        limit: int,
        where_clause: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synchronous search execution for thread pool."""
        return self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=limit,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

    async def delete_document(self, id: str) -> bool:
        """Remove a document from the ChromaDB collection."""
        if self.collection is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        # Run deletion in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        deleted = await loop.run_in_executor(
            None,
            self._delete_document_sync,
            id
        )

        if deleted:
            logger.debug(f"Deleted document with ID: {id}")
        return deleted

    def _delete_document_sync(self, id: str) -> bool:
        """Synchronous document deletion for thread pool execution."""
        try:
            # First check if document exists by querying for it
            result = self.collection.get(ids=[id])
            
            # If no IDs returned, document doesn't exist
            if not result["ids"] or len(result["ids"]) == 0:
                return False
            
            # Document exists, delete it
            self.collection.delete(ids=[id])
            return True
            
        except Exception as e:
            logger.warning(f"Failed to delete document {id}: {e}")
            return False

    async def close(self) -> None:
        """Clean up ChromaDB client and embedding provider resources."""
        if self.embedding_provider:
            await self.embedding_provider.close()
            
        if self.client:
            # ChromaDB client doesn't have explicit close method
            self.client = None
            
        self.collection = None
        logger.debug("ChromaDB provider resources cleaned up")
