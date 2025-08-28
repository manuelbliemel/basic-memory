"""Integration tests for vector search functionality."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from pathlib import Path

from basic_memory.services.search_service import SearchService
from basic_memory.schemas.search import SearchQuery
from basic_memory.models import Entity
from basic_memory.vector.config import SearchStrategy, EmbeddingModelType


class TestVectorSearchIntegration:
    """Integration tests for the complete vector search system."""

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
    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    @patch('sentence_transformers.SentenceTransformer')
    async def test_end_to_end_vector_search_flow(self, mock_sentence_transformer, mock_settings, mock_persistent_client, mock_config_manager):
        """Test complete end-to-end vector search flow."""
        # Mock configuration
        mock_config = Mock()
        mock_config.search_config.strategy = SearchStrategy.VECTOR_ONLY
        mock_config.search_config.vector.enabled = True
        mock_config.search_config.vector.embedding_model = EmbeddingModelType.LOCAL
        mock_config.search_config.vector.embedding_model_name = "all-MiniLM-L6-v2"
        mock_config.search_config.vector.provider = "chromadb"
        mock_config.search_config.vector.chunk_size = 500
        mock_config.search_config.vector.chunk_overlap = 50
        mock_config.search_config.vector.similarity_threshold = 0.3
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3, 0.4]
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model
        
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = mock_collection
        
        # Mock ChromaDB search results
        mock_search_results = {
            "ids": [["test-entity"]],
            "documents": [["Test document content"]],
            "metadatas": [[{
                "entity_id": "1", "project_id": "1", "type": "entity",
                "title": "Test Entity", "permalink": "test-entity"
            }]],
            "distances": [[0.2]]  # Similarity = 0.8
        }
        mock_collection.query.return_value = mock_search_results
        
        # Execute the complete flow
        query = SearchQuery(text="test search query")
        results = await self.search_service.search(query)
        
        # Verify initialization occurred
        assert self.search_service._vector_initialized is True
        
        # Verify search was executed
        assert len(results) == 1
        assert results[0].title == "Test Entity"
        assert results[0].score == 0.9
        
        # Verify components were called correctly
        mock_model.encode.assert_called()  # Embedding generation
        mock_collection.query.assert_called_once()  # Vector search

    @patch('basic_memory.services.search_service.ConfigManager')
    async def test_dual_indexing_integration(self, mock_config_manager):
        """Test that dual indexing works for both FTS5 and vector indices."""
        # Mock configuration with vector enabled
        mock_config = Mock()
        mock_config.search_config.vector.enabled = True
        mock_config.search_config.vector.embedding_model = EmbeddingModelType.LOCAL
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Mock entity
        mock_entity = Mock()
        mock_entity.id = 1
        mock_entity.is_markdown = True
        mock_entity.title = "Test Entity"
        mock_entity.permalink = "test-entity"
