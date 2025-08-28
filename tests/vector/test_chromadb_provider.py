"""Tests for ChromaDB vector search provider."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from basic_memory.vector.providers.chromadb_provider import ChromaDBProvider
from basic_memory.vector.embeddings.local_embedding import LocalEmbeddingProvider


class TestChromaDBProvider:
    """Test suite for ChromaDBProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock embedding provider
        self.mock_embedding_provider = Mock()
        self.mock_embedding_provider.initialize = AsyncMock()
        self.mock_embedding_provider.close = AsyncMock()
        self.mock_embedding_provider.get_dimension.return_value = 384
        self.mock_embedding_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_initialization_success(self, mock_settings, mock_persistent_client):
        """Test successful initialization of ChromaDB provider."""
        # Mock ChromaDB components
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        
        provider = ChromaDBProvider(
            embedding_provider=self.mock_embedding_provider,
            persist_directory=Path("/test/chroma")
        )
        
        await provider.initialize()
        
        assert provider.client is mock_client
        assert provider.collection is mock_collection
        self.mock_embedding_provider.initialize.assert_called_once()

    async def test_initialization_missing_dependency(self):
        """Test initialization failure when ChromaDB is not installed."""
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        
        # Mock the import to raise ImportError
        import sys
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == 'chromadb':
                raise ImportError("No module named 'chromadb'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(RuntimeError, match="chromadb package is required"):
                await provider.initialize()

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_initialization_existing_collection(self, mock_settings, mock_persistent_client):
        """Test initialization with existing collection."""
        # Mock ChromaDB components
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.get_collection.return_value = mock_collection  # Collection exists
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Should use existing collection, not create new one
        mock_client.get_collection.assert_called_once_with(name="basic_memory")
        mock_client.create_collection.assert_not_called()

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_index_document_with_embeddings(self, mock_settings, mock_persistent_client):
        """Test document indexing with pre-computed embeddings."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Test document indexing
        await provider.index_document(
            id="test-doc",
            content="Test content",
            metadata={"type": "entity", "title": "Test"},
            embeddings=[0.4, 0.5, 0.6]
        )
        
        # Should call ChromaDB upsert
        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert call_args[1]["ids"] == ["test-doc"]
        assert call_args[1]["documents"] == ["Test content"]
        assert call_args[1]["embeddings"] == [[0.4, 0.5, 0.6]]
        assert call_args[1]["metadatas"] == [{"type": "entity", "title": "Test"}]

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_index_document_without_embeddings(self, mock_settings, mock_persistent_client):
        """Test document indexing without pre-computed embeddings."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Test document indexing
        await provider.index_document(
            id="test-doc",
            content="Test content",
            metadata={"type": "entity"}
        )
        
        # Should generate embeddings and call ChromaDB upsert
        self.mock_embedding_provider.embed_text.assert_called_once_with("Test content")
        mock_collection.upsert.assert_called_once()

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_search_basic(self, mock_settings, mock_persistent_client):
        """Test basic vector search functionality."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        # Mock search results
        mock_search_results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [[{"title": "Doc 1"}, {"title": "Doc 2"}]],
            "distances": [[0.1, 0.3]]
        }
        mock_collection.query.return_value = mock_search_results
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Perform search
        results = await provider.search(
            query_embeddings=[0.1, 0.2, 0.3],
            limit=10,
            threshold=0.5
        )
        
        # Check results
        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "Content 1"
        assert results[0]["score"] == 0.95  # 1 - 0.1/2.0 distance
        assert results[1]["id"] == "doc2"
        assert results[1]["score"] == 0.85  # 1 - 0.3/2.0 distance

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_search_with_threshold_filtering(self, mock_settings, mock_persistent_client):
        """Test search with similarity threshold filtering."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        # Mock search results with varying similarity scores
        mock_search_results = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [["Content 1", "Content 2", "Content 3"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.4, 0.6]]  # Similarities: 0.9, 0.6, 0.4
        }
        mock_collection.query.return_value = mock_search_results
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Search with threshold 0.5
        results = await provider.search(
            query_embeddings=[0.1, 0.2, 0.3],
            threshold=0.5
        )
        
        # Should only return results above threshold
        assert len(results) == 3  # doc1 (0.95), doc2 (0.85), doc3 (0.7) - all above 0.5
        assert results[0]["id"] == "doc1"
        assert results[1]["id"] == "doc2"

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_search_with_metadata_filter(self, mock_settings, mock_persistent_client):
        """Test search with metadata filtering."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        mock_collection.query.return_value = {
            "ids": [["doc1"]], "documents": [["Content"]], 
            "metadatas": [[{}]], "distances": [[0.1]]
        }
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Search with metadata filter
        await provider.search(
            query_embeddings=[0.1, 0.2, 0.3],
            metadata_filter={"type": "entity", "project_id": "123"}
        )
        
        # Check that query was called with proper where clause
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args[1]
        expected_where = {
            "type": {"$eq": "entity"},
            "project_id": {"$eq": "123"}
        }
        assert call_args["where"] == expected_where


    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_delete_document_error_handling(self, mock_settings, mock_persistent_client):
        """Test graceful error handling during document deletion."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_collection.get.side_effect = Exception("Get failed")
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Should not raise exception, but handle gracefully
        result = await provider.delete_document("test-doc")
        
        # Should return False due to exception
        assert result is False

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_close(self, mock_settings, mock_persistent_client):
        """Test provider cleanup."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        await provider.close()
        
        # Should clean up embedding provider
        self.mock_embedding_provider.close.assert_called_once()
        assert provider.client is None
        assert provider.collection is None

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_metadata_processing(self, mock_settings, mock_persistent_client):
        """Test that metadata values are converted to strings for ChromaDB."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Index document with mixed metadata types
        metadata = {
            "string_field": "test",
            "int_field": 123,
            "bool_field": True,
            "none_field": None
        }
        
        await provider.index_document(
            id="test-doc",
            content="Test",
            metadata=metadata,
            embeddings=[0.1, 0.2, 0.3]
        )
        
        # Check that metadata was processed correctly
        call_args = mock_collection.upsert.call_args[1]
        processed_metadata = call_args["metadatas"][0]
        
        assert processed_metadata["string_field"] == "test"
        assert processed_metadata["int_field"] == "123"  # Converted to string
        assert processed_metadata["bool_field"] == "True"  # Converted to string
        assert "none_field" not in processed_metadata  # None values filtered out

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_search_empty_results(self, mock_settings, mock_persistent_client):
        """Test search with no results."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        # Mock empty search results
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        results = await provider.search(query_embeddings=[0.1, 0.2, 0.3])
        
        assert results == []

    async def test_not_initialized_errors(self):
        """Test that methods fail when provider is not initialized."""
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await provider.index_document("id", "content", {})
        
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await provider.search([0.1, 0.2])
        
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await provider.delete_document("id")

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_default_persist_directory(self, mock_settings, mock_persistent_client):
        """Test that default persist directory is set correctly."""
        mock_client = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = Mock()
        mock_client.get_collection.side_effect = Exception("Not found")
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Should use default directory
        expected_path = str(Path.home() / ".basic-memory" / "chroma")
        mock_persistent_client.assert_called_once()
        call_args = mock_persistent_client.call_args[1]
        assert call_args["path"] == expected_path

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_custom_collection_name(self, mock_settings, mock_persistent_client):
        """Test using custom collection name."""
        mock_client = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = Mock()
        mock_client.get_collection.side_effect = Exception("Not found")
        
        provider = ChromaDBProvider(
            embedding_provider=self.mock_embedding_provider,
            collection_name="custom_collection"
        )
        await provider.initialize()
        
        # Should use custom collection name
        mock_client.get_collection.assert_called_with(name="custom_collection")
        mock_client.create_collection.assert_called_with(
            name="custom_collection",
            metadata={"embedding_dimension": 384}
        )

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_delete_document(self, mock_settings, mock_persistent_client):
        """Test successful document deletion."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        # Mock get() to return document exists
        mock_collection.get.return_value = {"ids": ["test-doc"]}
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Delete document
        result = await provider.delete_document("test-doc")
        
        # Should check existence first, then delete
        mock_collection.get.assert_called_once_with(ids=["test-doc"])
        mock_collection.delete.assert_called_once_with(ids=["test-doc"])
        assert result is True

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_delete_document_not_exists(self, mock_settings, mock_persistent_client):
        """Test deletion when document doesn't exist."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        
        # Mock get() to return no documents
        mock_collection.get.return_value = {"ids": []}
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Delete non-existent document
        result = await provider.delete_document("non-existent")
        
        # Should check existence, but not call delete
        mock_collection.get.assert_called_once_with(ids=["non-existent"])
        mock_collection.delete.assert_not_called()
        assert result is False

    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    async def test_delete_document_error_handling(self, mock_settings, mock_persistent_client):
        """Test graceful error handling during document deletion."""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_collection.get.side_effect = Exception("Get failed")
        
        provider = ChromaDBProvider(embedding_provider=self.mock_embedding_provider)
        await provider.initialize()
        
        # Should not raise exception, but handle gracefully
        result = await provider.delete_document("test-doc")
        
        # Should return False due to exception
        assert result is False
