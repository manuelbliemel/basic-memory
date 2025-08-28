"""OpenAI embedding provider using OpenAI API."""

from typing import List, Optional

from loguru import logger

from .base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using OpenAI API."""

    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.client = None

    async def _initialize_impl(self) -> None:
        """Initialize the OpenAI client."""
        try:
            # Import here to avoid dependency issues if not installed
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError(
                "openai package is required for OpenAI embeddings. "
                "Install it with: pip install openai"
            )

        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key is required. Set it in the configuration or environment variable OPENAI_API_KEY"
            )

        logger.info(f"Initializing OpenAI embedding provider with model: {self.model_name}")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Set embedding dimension based on model
        if "ada-002" in self.model_name:
            self._dimension = 1536
        elif "3-small" in self.model_name:
            self._dimension = 1536  
        elif "3-large" in self.model_name:
            self._dimension = 3072
        else:
            # Default dimension, may need to be updated for new models
            self._dimension = 1536
            logger.warning(f"Unknown model {self.model_name}, using default dimension {self._dimension}")

        logger.info(f"OpenAI embedding provider initialized. Embedding dimension: {self._dimension}")

    async def _close_impl(self) -> None:
        """Clean up OpenAI client resources."""
        if self.client is not None:
            await self.client.close()
            self.client = None
            logger.debug("OpenAI client resources cleaned up")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text using OpenAI API."""
        if not self._initialized or self.client is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._dimension

        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently using batch API."""
        if not self._initialized or self.client is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        if not texts:
            return []

        # Filter out empty texts and track their positions
        non_empty_texts = []
        empty_indices = []
        text_mapping = {}  # Maps original index to filtered index
        
        for i, text in enumerate(texts):
            if text.strip():
                text_mapping[i] = len(non_empty_texts)
                non_empty_texts.append(text)
            else:
                empty_indices.append(i)

        if not non_empty_texts:
            # All texts are empty
            return [[0.0] * self._dimension for _ in texts]

        try:
            # Use batch API for efficiency
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=non_empty_texts
            )

            # Build result list with proper ordering
            results = []
            for i in range(len(texts)):
                if i in empty_indices:
                    results.append([0.0] * self._dimension)
                else:
                    filtered_idx = text_mapping[i]
                    results.append(response.data[filtered_idx].embedding)

            return results
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise
