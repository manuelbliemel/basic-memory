"""Tests for embedding providers."""

import pytest
import sys
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from basic_memory.vector.embeddings.base import BaseEmbeddingProvider
from basic_memory.vector.embeddings.local_embedding import LocalEmbeddingProvider
from basic_memory.vector.embeddings.openai_embedding import OpenAIEmbeddingProvider


class TestLocalEmbeddingProvider:
    """Test suite for LocalEmbeddingProvider."""

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test successful initialization of local embedding provider."""
        # Mock the sentence transformer at import time
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer = Mock(return_value=mock_model)
        
        with patch.dict('sys.modules', {'sentence_transformers': Mock(SentenceTransformer=mock_sentence_transformer)}):
            provider = LocalEmbeddingProvider("all-MiniLM-L6-v2")
            await provider.initialize()
            
            assert provider._initialized is True
            assert provider._dimension == 384
            mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    async def test_initialization_missing_dependency(self):
        """Test initialization failure when sentence-transformers is not installed."""
        provider = LocalEmbeddingProvider("test-model")
        
        # Mock the import to raise ImportError
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == 'sentence_transformers':
                raise ImportError("No module named 'sentence_transformers'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(RuntimeError, match="sentence-transformers package is required"):
                await provider.initialize()

    @patch('sentence_transformers.SentenceTransformer')
    async def test_embed_text(self, mock_sentence_transformer):
        """Test single text embedding."""
        # Setup mock
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model
        
        provider = LocalEmbeddingProvider()
        await provider.initialize()
        
        result = await provider.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with("test text")

    @patch('sentence_transformers.SentenceTransformer')
    async def test_embed_empty_text(self, mock_sentence_transformer):
        """Test embedding empty text returns zero vector."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        provider = LocalEmbeddingProvider()
        await provider.initialize()
        
        result = await provider.embed_text("")
        
        assert result == [0.0] * 384
        mock_model.encode.assert_not_called()

    @patch('sentence_transformers.SentenceTransformer')
    async def test_embed_texts_batch(self, mock_sentence_transformer):
        """Test batch text embedding."""
        # Setup mock
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # Create mock embeddings that can be subscripted
        mock_embedding1 = Mock()
        mock_embedding1.tolist.return_value = [0.1, 0.2]
        mock_embedding2 = Mock()
        mock_embedding2.tolist.return_value = [0.3, 0.4]
        
        mock_embeddings = [mock_embedding1, mock_embedding2]
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        provider = LocalEmbeddingProvider()
        await provider.initialize()
        
        texts = ["text 1", "text 2"]
        result = await provider.embed_texts(texts)
        
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_model.encode.assert_called_once_with(texts)

    @patch('sentence_transformers.SentenceTransformer')
    async def test_embed_texts_with_empty_texts(self, mock_sentence_transformer):
        """Test batch embedding with some empty texts."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 2
        
        # Create mock embedding that can be subscripted  
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2]
        
        mock_embeddings = [mock_embedding]  # Only one embedding for non-empty text
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        provider = LocalEmbeddingProvider()
        await provider.initialize()
        
        texts = ["valid text", "", "  \t  "]  # Mix of valid and empty
        result = await provider.embed_texts(texts)
        
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]  # Valid embedding
        assert result[1] == [0.0, 0.0]  # Zero vector for empty
        assert result[2] == [0.0, 0.0]  # Zero vector for whitespace-only

    async def test_not_initialized_error(self):
        """Test that methods fail when provider is not initialized."""
        provider = LocalEmbeddingProvider()
        
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await provider.embed_text("test")
        
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await provider.embed_texts(["test"])
        
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            provider.get_dimension()

    @patch('sentence_transformers.SentenceTransformer')
    async def test_close(self, mock_sentence_transformer):
        """Test provider cleanup."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        provider = LocalEmbeddingProvider()
        await provider.initialize()
        
        assert provider._initialized is True
        
        await provider.close()
        
        assert provider._initialized is False
        assert provider.model is None


class TestOpenAIEmbeddingProvider:
    """Test suite for OpenAIEmbeddingProvider."""

    @patch('openai.AsyncOpenAI')
    async def test_initialization_success(self, mock_openai):
        """Test successful initialization of OpenAI embedding provider."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        await provider.initialize()
        
        assert provider._initialized is True
        assert provider._dimension == 1536  # Default for ada-002
        mock_openai.assert_called_once_with(api_key="test-key")

    async def test_initialization_missing_dependency(self):
        """Test initialization failure when openai package is not installed."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        
        # Mock the import to raise ImportError
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == 'openai':
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(RuntimeError, match="openai package is required"):
                await provider.initialize()

    async def test_initialization_missing_api_key(self):
        """Test initialization failure when API key is missing."""
        provider = OpenAIEmbeddingProvider()  # No API key
        
        with pytest.raises(RuntimeError, match="OpenAI API key is required"):
            await provider.initialize()

    @patch('openai.AsyncOpenAI')
    async def test_embed_text(self, mock_openai):
        """Test single text embedding via OpenAI API."""
        # Setup mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        await provider.initialize()
        
        result = await provider.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input="test text"
        )

    @patch('openai.AsyncOpenAI')
    async def test_embed_empty_text(self, mock_openai):
        """Test embedding empty text returns zero vector."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        await provider.initialize()
        
        result = await provider.embed_text("")
        
        assert result == [0.0] * 1536
        mock_client.embeddings.create.assert_not_called()

    @patch('openai.AsyncOpenAI')
    async def test_embed_texts_batch(self, mock_openai):
        """Test batch text embedding via OpenAI API."""
        # Setup mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(), Mock()]
        mock_response.data[0].embedding = [0.1, 0.2]
        mock_response.data[1].embedding = [0.3, 0.4]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        await provider.initialize()
        
        texts = ["text 1", "text 2"]
        result = await provider.embed_texts(texts)
        
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=texts
        )

    @patch('openai.AsyncOpenAI')
    async def test_embed_texts_with_empty_texts(self, mock_openai):
        """Test batch embedding with some empty texts."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock()]  # Only one embedding for non-empty text
        mock_response.data[0].embedding = [0.1, 0.2]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key", model_name="text-embedding-ada-002")
        await provider.initialize()
        
        texts = ["valid text", "", "  \t  "]  # Mix of valid and empty
        result = await provider.embed_texts(texts)
        
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]  # Valid embedding
        assert result[1] == [0.0] * 1536  # Zero vector for empty
        assert result[2] == [0.0] * 1536  # Zero vector for whitespace-only
        
        # Should only call API with non-empty text
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=["valid text"]
        )

    @patch('openai.AsyncOpenAI')
    async def test_api_error_handling(self, mock_openai):
        """Test handling of OpenAI API errors."""
        mock_client = Mock()
        mock_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        await provider.initialize()
        
        with pytest.raises(Exception, match="API Error"):
            await provider.embed_text("test text")

    @patch('openai.AsyncOpenAI')
    async def test_model_dimension_detection(self, mock_openai):
        """Test that different models get correct dimensions."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Test ada-002 model
        provider = OpenAIEmbeddingProvider(api_key="test-key", model_name="text-embedding-ada-002")
        await provider.initialize()
        assert provider._dimension == 1536
        
        # Test 3-small model
        provider = OpenAIEmbeddingProvider(api_key="test-key", model_name="text-embedding-3-small")
        await provider.initialize()
        assert provider._dimension == 1536
        
        # Test 3-large model
        provider = OpenAIEmbeddingProvider(api_key="test-key", model_name="text-embedding-3-large")
        await provider.initialize()
        assert provider._dimension == 3072
        
        # Test unknown model (should use default)
        provider = OpenAIEmbeddingProvider(api_key="test-key", model_name="unknown-model")
        await provider.initialize()
        assert provider._dimension == 1536

    @patch('openai.AsyncOpenAI')
    async def test_close(self, mock_openai):
        """Test provider cleanup."""
        mock_client = Mock()
        mock_client.close = AsyncMock()
        mock_openai.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        await provider.initialize()
        
        assert provider._initialized is True
        
        await provider.close()
        
        assert provider._initialized is False
        assert provider.client is None
        mock_client.close.assert_called_once()


class TestBaseEmbeddingProvider:
    """Test suite for BaseEmbeddingProvider."""

    class MockEmbeddingProvider(BaseEmbeddingProvider):
        """Mock implementation for testing base functionality."""
        
        async def _initialize_impl(self):
            self._dimension = 100
            
        async def _close_impl(self):
            pass
            
        async def embed_text(self, text: str):
            if not self._initialized:
                raise RuntimeError("Provider not initialized. Call initialize() first.")
            return [0.1, 0.2, 0.3]

    async def test_initialization_lifecycle(self):
        """Test initialization lifecycle management."""
        provider = self.MockEmbeddingProvider()
        
        assert provider._initialized is False
        
        # Initialize
        await provider.initialize()
        assert provider._initialized is True
        assert provider._dimension == 100
        
        # Initialize again (should be no-op)
        await provider.initialize()
        assert provider._initialized is True

    async def test_close_lifecycle(self):
        """Test close lifecycle management."""
        provider = self.MockEmbeddingProvider()
        
        await provider.initialize()
        assert provider._initialized is True
        
        await provider.close()
        assert provider._initialized is False
        
        # Close again (should be no-op)
        await provider.close()
        assert provider._initialized is False

    async def test_get_dimension_before_init(self):
        """Test that get_dimension fails before initialization."""
        provider = self.MockEmbeddingProvider()
        
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            provider.get_dimension()

    async def test_get_dimension_after_init(self):
        """Test get_dimension after initialization."""
        provider = self.MockEmbeddingProvider()
        await provider.initialize()
        
        assert provider.get_dimension() == 100

    async def test_default_embed_texts_implementation(self):
        """Test that default embed_texts calls embed_text for each text."""
        provider = self.MockEmbeddingProvider()
        await provider.initialize()
        
        texts = ["text1", "text2", "text3"]
        results = await provider.embed_texts(texts)
        
        assert len(results) == 3
        assert all(result == [0.1, 0.2, 0.3] for result in results)

    async def test_embed_texts_empty_list(self):
        """Test embed_texts with empty list."""
        provider = self.MockEmbeddingProvider()
        await provider.initialize()
        
        results = await provider.embed_texts([])
        
        assert results == []
