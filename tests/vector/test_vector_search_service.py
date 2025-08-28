"""Tests for SearchService vector search integration."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from basic_memory.services.search_service import SearchService
from basic_memory.schemas.search import SearchQuery, SearchItemType
from basic_memory.repository.search_repository import SearchIndexRow
from basic_memory.vector.config import SearchStrategy, EmbeddingModelType


class TestSearchServiceVectorIntegration:
    """Test suite for SearchService vector search integration."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock repositories and services
        self.mock_search_repository = Mock()
        self.mock_entity_repository = Mock()
        self.mock_file_service = Mock()
        
        # Create SearchService instance
        self.search_service = SearchService(
            search_repository=self.mock_search_repository,
            entity_repository=self.mock_entity_repository,
            file_service=self.mock_file_service
        )

    @patch('basic_memory.services.search_service.ConfigManager')
    async def test_init_vector_search_disabled(self, mock_config_manager):
        """Test vector search initialization when disabled in config."""
        # Mock configuration with vector search disabled
        mock_config = Mock()
        mock_config.search_config.vector.enabled = False
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        await self.search_service.init_vector_search()
        
        assert self.search_service._vector_initialized is False
        assert self.search_service.vector_provider is None

    @patch('basic_memory.services.search_service.ConfigManager')
    @patch('basic_memory.services.search_service.LocalEmbeddingProvider')
    @patch('basic_memory.services.search_service.ChromaDBProvider')
    async def test_init_vector_search_local_embeddings(self, mock_chromadb, mock_local_embedding, mock_config_manager):
        """Test vector search initialization with local embeddings."""
        # Mock configuration
        mock_config = Mock()
        mock_config.search_config.vector.enabled = True
        mock_config.search_config.vector.embedding_model = EmbeddingModelType.LOCAL
        mock_config.search_config.vector.embedding_model_name = "test-model"
        mock_config.search_config.vector.provider = "chromadb"
        mock_config.search_config.vector.chunk_size = 500
        mock_config.search_config.vector.chunk_overlap = 50
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock providers
        mock_embedding_instance = Mock()
        mock_embedding_instance.initialize = AsyncMock()
        mock_local_embedding.return_value = mock_embedding_instance
        
        mock_vector_instance = Mock()
        mock_vector_instance.initialize = AsyncMock()
        mock_chromadb.return_value = mock_vector_instance
        
        await self.search_service.init_vector_search()
        
        assert self.search_service._vector_initialized is True
        assert self.search_service.embedding_provider is mock_embedding_instance
        assert self.search_service.vector_provider is mock_vector_instance
        assert self.search_service.text_chunker is not None

    @patch('basic_memory.services.search_service.ConfigManager')
    @patch('basic_memory.services.search_service.OpenAIEmbeddingProvider')
    @patch('basic_memory.services.search_service.ChromaDBProvider')
    async def test_init_vector_search_openai_embeddings(self, mock_chromadb, mock_openai_embedding, mock_config_manager):
        """Test vector search initialization with OpenAI embeddings."""
        # Mock configuration
        mock_config = Mock()
        mock_config.search_config.vector.enabled = True
        mock_config.search_config.vector.embedding_model = EmbeddingModelType.OPENAI
        mock_config.search_config.vector.embedding_model_name = "text-embedding-ada-002"
        mock_config.search_config.vector.api_key = "test-api-key"
        mock_config.search_config.vector.provider = "chromadb"
        mock_config.search_config.vector.chunk_size = 500
        mock_config.search_config.vector.chunk_overlap = 50
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock providers
        mock_embedding_instance = Mock()
        mock_embedding_instance.initialize = AsyncMock()
        mock_openai_embedding.return_value = mock_embedding_instance
        
        mock_vector_instance = Mock()
        mock_vector_instance.initialize = AsyncMock()
        mock_chromadb.return_value = mock_vector_instance
        
        await self.search_service.init_vector_search()
        
        assert self.search_service._vector_initialized is True
        mock_openai_embedding.assert_called_once_with(
            model_name="text-embedding-ada-002",
            api_key="test-api-key"
        )

    @patch('basic_memory.services.search_service.ConfigManager')
    async def test_search_fuzzy_only_strategy(self, mock_config_manager):
        """Test search with fuzzy_only strategy."""
        # Mock configuration
        mock_config = Mock()
        mock_config.search_config.strategy = SearchStrategy.FUZZY_ONLY
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock fuzzy search results
        expected_results = [
            SearchIndexRow(
                id=1, project_id=1, type="entity", file_path="/test.md",
                title="Test", permalink="test", created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        self.mock_search_repository.search = AsyncMock(return_value=expected_results)
        
        query = SearchQuery(text="test query")
        results = await self.search_service.search(query)
        
        assert results == expected_results
        self.mock_search_repository.search.assert_called_once()

    @patch('basic_memory.services.search_service.ConfigManager')
    async def test_search_vector_only_strategy(self, mock_config_manager):
        """Test search with vector_only strategy."""
        # Mock configuration
        mock_config = Mock()
        mock_config.search_config.strategy = SearchStrategy.VECTOR_ONLY
        mock_config.search_config.vector.enabled = True
        mock_config.search_config.vector.similarity_threshold = 0.3
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock vector search components
        self.search_service._vector_initialized = True
        self.search_service.vector_provider = Mock()
        self.search_service.embedding_provider = Mock()
        self.search_service.embedding_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        # Mock vector search results
        vector_results = [
            {
                "id": "test-1",
                "content": "Test content",
                "metadata": {
                    "entity_id": "1",
                    "project_id": "1",
                    "type": "entity",
                    "title": "Test",
                    "permalink": "test"
                },
                "score": 0.8
            }
        ]
        self.search_service.vector_provider.search = AsyncMock(return_value=vector_results)
        
        query = SearchQuery(text="test query")
        results = await self.search_service.search(query)
        
        assert len(results) == 1
        assert results[0].title == "Test"
        self.search_service.embedding_provider.embed_text.assert_called_once_with("test query")

    @patch('basic_memory.services.search_service.ConfigManager')
    async def test_search_hybrid_strategy(self, mock_config_manager):
        """Test search with hybrid strategy."""
        # Mock configuration
        mock_config = Mock()
        mock_config.search_config.strategy = SearchStrategy.HYBRID
        mock_config.search_config.vector.enabled = True
        mock_config.search_config.vector.similarity_threshold = 0.3
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock fuzzy search results
        fuzzy_results = [
            SearchIndexRow(
                id=1, project_id=1, type="entity", file_path="/test1.md",
                title="Fuzzy Result", permalink="fuzzy-result",
                created_at=datetime.now(), updated_at=datetime.now()
            )
        ]
        self.mock_search_repository.search = AsyncMock(return_value=fuzzy_results)
        
        # Mock vector search components
        self.search_service._vector_initialized = True
        self.search_service.vector_provider = Mock()
        self.search_service.embedding_provider = Mock()
        self.search_service.embedding_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        vector_results = [
            {
                "id": "test-2",
                "content": "Vector content",
                "metadata": {
                    "entity_id": "2",
                    "project_id": "1",
                    "type": "entity",
                    "title": "Vector Result",
                    "permalink": "vector-result"
                },
                "score": 0.8
            }
        ]
        self.search_service.vector_provider.search = AsyncMock(return_value=vector_results)
        
        query = SearchQuery(text="test query")
        results = await self.search_service.search(query)
        
        # Should combine both results
        assert len(results) >= 1  # At least one result (combination logic tested separately)
        
        # Both searches should have been called
        self.mock_search_repository.search.assert_called_once()
        self.search_service.vector_provider.search.assert_called_once()

    @patch('basic_memory.services.search_service.ConfigManager')
    async def test_search_permalink_fallback_to_fuzzy(self, mock_config_manager):
        """Test that permalink searches always use fuzzy search regardless of strategy."""
        # Mock configuration with vector strategy
        mock_config = Mock()
        mock_config.search_config.strategy = SearchStrategy.VECTOR_ONLY
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock fuzzy search results
        expected_results = [
            SearchIndexRow(
                id=1, project_id=1, type="entity", file_path="/test.md",
                title="Test", permalink="test-permalink",
                created_at=datetime.now(), updated_at=datetime.now()
            )
        ]
        self.mock_search_repository.search = AsyncMock(return_value=expected_results)
        
        # Search by permalink
        query = SearchQuery(permalink="test-permalink")
        results = await self.search_service.search(query)
        
        assert results == expected_results
        # Should call fuzzy search, not vector search
        self.mock_search_repository.search.assert_called_once()

    async def test_search_no_criteria(self):
        """Test search with no criteria returns empty results."""
        query = SearchQuery()  # No text, permalink, etc.
        results = await self.search_service.search(query)
        
        assert results == []

    @patch('basic_memory.services.search_service.ConfigManager')
    async def test_vector_search_fallback_on_failure(self, mock_config_manager):
        """Test that vector search failures fall back to fuzzy search."""
        # Mock configuration
        mock_config = Mock()
        mock_config.search_config.strategy = SearchStrategy.VECTOR_ONLY
        mock_config.search_config.vector.enabled = True
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock vector search components with failure
        self.search_service._vector_initialized = True
        self.search_service.vector_provider = Mock()
        self.search_service.embedding_provider = Mock()
        self.search_service.embedding_provider.embed_text = AsyncMock(side_effect=Exception("Embedding failed"))
        
        # Mock fallback fuzzy results
        fallback_results = [
            SearchIndexRow(
                id=1, project_id=1, type="entity", file_path="/test.md",
                title="Fallback", permalink="fallback",
                created_at=datetime.now(), updated_at=datetime.now()
            )
        ]
        self.mock_search_repository.search = AsyncMock(return_value=fallback_results)
        
        query = SearchQuery(text="test query")
        results = await self.search_service.search(query)
        
        # Should fall back to fuzzy search
        assert results == fallback_results
        self.mock_search_repository.search.assert_called_once()

    async def test_index_entity_vector_success(self):
        """Test successful vector indexing of an entity."""
        # Mock entity
        mock_entity = Mock()
        mock_entity.id = 1
        mock_entity.project_id = 1
        mock_entity.title = "Test Entity"
        mock_entity.permalink = "test-entity"
        mock_entity.file_path = "/test.md"
        mock_entity.entity_type = "note"
        
        # Mock file content
        self.mock_file_service.read_entity_content = AsyncMock(return_value="Test content for indexing")
        
        # Mock vector components
        self.search_service._vector_initialized = True
        self.search_service.vector_provider = Mock()
        self.search_service.vector_provider.index_document = AsyncMock()
        self.search_service.text_chunker = Mock()
        self.search_service.text_chunker.chunk_size = 500
        
        await self.search_service.index_entity_vector(mock_entity)
        
        # Should index the document
        self.search_service.vector_provider.index_document.assert_called_once()
        call_args = self.search_service.vector_provider.index_document.call_args[1]
        assert call_args["id"] == "test-entity"
        assert call_args["content"] == "Test content for indexing"
        assert call_args["metadata"]["entity_id"] == "1"

    async def test_index_entity_vector_with_chunking(self):
        """Test vector indexing with content chunking."""
        # Mock entity
        mock_entity = Mock()
        mock_entity.id = 1
        mock_entity.project_id = 1
        mock_entity.title = "Test Entity"
        mock_entity.permalink = "test-entity"
        mock_entity.file_path = "/test.md"
        mock_entity.entity_type = "note"
        
        # Mock long content that needs chunking
        long_content = "A" * 1000  # Content longer than chunk size
        self.mock_file_service.read_entity_content = AsyncMock(return_value=long_content)
        
        # Mock vector components
        self.search_service._vector_initialized = True
        self.search_service.vector_provider = Mock()
        self.search_service.vector_provider.index_document = AsyncMock()
        self.search_service.text_chunker = Mock()
        self.search_service.text_chunker.chunk_size = 500
        self.search_service.text_chunker.chunk_text.return_value = [
            {"content": "Chunk 1", "metadata": {"chunk_index": 0}},
            {"content": "Chunk 2", "metadata": {"chunk_index": 1}}
        ]
        
        await self.search_service.index_entity_vector(mock_entity)
        
        # Should chunk and index multiple documents
        assert self.search_service.vector_provider.index_document.call_count == 2
        self.search_service.text_chunker.chunk_text.assert_called_once()

    async def test_index_entity_vector_not_initialized(self):
        """Test vector indexing when vector search is not initialized."""
        mock_entity = Mock()
        self.search_service._vector_initialized = False
        
        # Should return early without error
        await self.search_service.index_entity_vector(mock_entity)
        
        # No file service calls should be made
        self.mock_file_service.read_entity_content.assert_not_called()

    async def test_delete_entity_vector_success(self):
        """Test successful vector deletion of an entity."""
        # Mock entity
        mock_entity = Mock()
        mock_entity.id = 1
        mock_entity.permalink = "test-entity"
        
        self.mock_entity_repository.find_by_id = AsyncMock(return_value=mock_entity)
        
        # Mock vector components with explicit boolean returns
        self.search_service._vector_initialized = True
        self.search_service.vector_provider = Mock()
        
        # Mock delete_document to return True for main doc and first 2 chunks, then False
        delete_results = [True, True, True, False]  # Main doc + 2 chunks + stop
        self.search_service.vector_provider.delete_document = AsyncMock(side_effect=delete_results)
        
        await self.search_service.delete_entity_vector(1)
        
        # Should delete main document and chunks until False is returned
        assert self.search_service.vector_provider.delete_document.call_count == 4
        
        # Check the calls made
        calls = self.search_service.vector_provider.delete_document.call_args_list
        assert calls[0][0][0] == "test-entity"  # Main document
        assert calls[1][0][0] == "test-entity#0"  # First chunk
        assert calls[2][0][0] == "test-entity#1"  # Second chunk  
        assert calls[3][0][0] == "test-entity#2"  # Third chunk (returns False)

    async def test_delete_entity_vector_not_initialized(self):
        """Test vector deletion when vector search is not initialized."""
        self.search_service._vector_initialized = False
        
        # Should return early without error
        await self.search_service.delete_entity_vector(1)
        
        # No repository calls should be made
        self.mock_entity_repository.find_by_id.assert_not_called()

    @patch('basic_memory.services.search_service.ConfigManager')
    async def test_search_strategy_fuzzy_primary_with_results(self, mock_config_manager):
        """Test fuzzy_primary strategy when fuzzy search has results."""
        # Mock configuration
        mock_config = Mock()
        mock_config.search_config.strategy = SearchStrategy.FUZZY_PRIMARY
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock fuzzy search with results
        fuzzy_results = [
            SearchIndexRow(
                id=1, project_id=1, type="entity", file_path="/test.md",
                title="Fuzzy Result", permalink="fuzzy",
                created_at=datetime.now(), updated_at=datetime.now()
            )
        ]
        self.mock_search_repository.search = AsyncMock(return_value=fuzzy_results)
        
        query = SearchQuery(text="test query")
        results = await self.search_service.search(query)
        
        # Should return fuzzy results without trying vector search
        assert results == fuzzy_results
        self.mock_search_repository.search.assert_called_once()

    @patch('basic_memory.services.search_service.ConfigManager')
    async def test_search_strategy_fuzzy_primary_fallback(self, mock_config_manager):
        """Test fuzzy_primary strategy falling back to vector search."""
        # Mock configuration
        mock_config = Mock()
        mock_config.search_config.strategy = SearchStrategy.FUZZY_PRIMARY
        mock_config.search_config.vector.enabled = True
        mock_config.search_config.vector.similarity_threshold = 0.3
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock fuzzy search with no results
        self.mock_search_repository.search = AsyncMock(return_value=[])
        
        # Mock vector search components
        self.search_service._vector_initialized = True
        self.search_service.vector_provider = Mock()
        self.search_service.embedding_provider = Mock()
        self.search_service.embedding_provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        vector_results = [
            {
                "id": "test-1",
                "content": "Vector content",
                "metadata": {
                    "entity_id": "1", "project_id": "1", "type": "entity",
                    "title": "Vector Result", "permalink": "vector"
                },
                "score": 0.8
            }
        ]
        self.search_service.vector_provider.search = AsyncMock(return_value=vector_results)
        
        query = SearchQuery(text="test query")
        results = await self.search_service.search(query)
        
        # Should fall back to vector search
        assert len(results) == 1
        assert results[0].title == "Vector Result"
        
        # Both searches should have been called
        self.mock_search_repository.search.assert_called_once()
        self.search_service.vector_provider.search.assert_called_once()

    async def test_index_entity_data_dual_indexing(self):
        """Test that index_entity_data handles both FTS5 and vector indexing."""
        # Mock entity
        mock_entity = Mock()
        mock_entity.id = 1
        mock_entity.is_markdown = True
        
        # Mock existing methods
        self.search_service.index_entity_markdown = AsyncMock()
        self.search_service.delete_entity_vector = AsyncMock()
        self.mock_search_repository.delete_by_entity_id = AsyncMock()
        
        # Mock vector initialization
        self.search_service._vector_initialized = True
        self.search_service.vector_provider = Mock()
        
        await self.search_service.index_entity_data(mock_entity)
        
        # Should delete from both indices
        self.mock_search_repository.delete_by_entity_id.assert_called_once_with(entity_id=1)
        self.search_service.delete_entity_vector.assert_called_once_with(1)
        
        # Should reindex
        self.search_service.index_entity_markdown.assert_called_once_with(mock_entity)

    async def test_vector_search_initialization_on_demand(self):
        """Test that vector search is initialized on-demand when needed."""
        # Mock repositories with proper AsyncMock
        mock_search_repo = Mock()
        mock_search_repo.search = AsyncMock(return_value=[])
        mock_entity_repo = Mock()
        mock_file_service = Mock()
        
        # Create a fresh service instance to test initialization
        search_service = SearchService(
            search_repository=mock_search_repo,
            entity_repository=mock_entity_repo,
            file_service=mock_file_service
        )
        
        # Mock the init_vector_search method
        search_service.init_vector_search = AsyncMock()
        
        with patch('basic_memory.services.search_service.ConfigManager') as mock_config_manager:
            mock_config = Mock()
            mock_config.search_config.strategy = SearchStrategy.VECTOR_ONLY
            mock_config_manager.return_value.load_config.return_value = mock_config
            
            query = SearchQuery(text="test query")
            await search_service.search(query)
            
            # Should have attempted to initialize vector search
            search_service.init_vector_search.assert_called_once()
