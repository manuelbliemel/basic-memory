"""Tests for text chunking functionality."""

import pytest

from basic_memory.vector.chunker import TextChunker


class TestTextChunker:
    """Test suite for TextChunker."""

    def test_init_with_valid_parameters(self):
        """Test chunker initialization with valid parameters."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_init_with_invalid_chunk_size(self):
        """Test chunker initialization with invalid chunk size."""
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            TextChunker(chunk_size=0)
        
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            TextChunker(chunk_size=-10)

    def test_init_with_invalid_overlap(self):
        """Test chunker initialization with invalid overlap."""
        with pytest.raises(ValueError, match="Chunk overlap cannot be negative"):
            TextChunker(chunk_size=100, chunk_overlap=-5)
        
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_chunk_empty_text(self):
        """Test chunking empty or whitespace-only text."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        
        # Empty string
        assert chunker.chunk_text("") == []
        
        # Whitespace only
        assert chunker.chunk_text("   \n\t  ") == []

    def test_chunk_small_text(self):
        """Test chunking text smaller than chunk size."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a small text."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["total_chunks"] == 1
        assert chunks[0]["metadata"]["chunk_length"] == len(text)

    def test_chunk_large_text(self):
        """Test chunking text larger than chunk size."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a longer text that should be split into multiple chunks. " \
               "It contains several sentences to test the chunking functionality properly."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        
        # Check metadata consistency
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["total_chunks"] == len(chunks)
            assert "chunk_length" in chunk["metadata"]
            assert len(chunk["content"]) > 0

        # Check overlap - subsequent chunks should have some common content
        if len(chunks) > 1:
            # The overlap should be reflected in the positioning
            for i in range(len(chunks) - 1):
                current_end = chunks[i]["metadata"]["chunk_end"]
                next_start = chunks[i + 1]["metadata"]["chunk_start"]
                # Next chunk should start before current ends (overlap)
                assert next_start < current_end

    def test_chunk_with_natural_breaks(self):
        """Test that chunker prefers natural break points."""
        chunker = TextChunker(chunk_size=80, chunk_overlap=10)
        text = "First paragraph.\n\nSecond paragraph with more content that goes beyond the chunk size limit."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        # Should break at paragraph boundary if possible
        if len(chunks) > 1:
            first_chunk = chunks[0]["content"]
            # Should end near paragraph break
            assert "First paragraph" in first_chunk

    def test_chunk_with_metadata(self):
        """Test chunking with additional metadata."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        text = "This is a test text that will be chunked with metadata."
        metadata = {"entity_id": "123", "title": "Test Entity"}
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) > 0
        for chunk in chunks:
            # Original metadata should be preserved
            assert chunk["metadata"]["entity_id"] == "123"
            assert chunk["metadata"]["title"] == "Test Entity"
            # Chunk-specific metadata should be added
            assert "chunk_index" in chunk["metadata"]
            assert "total_chunks" in chunk["metadata"]

    def test_chunk_documents(self):
        """Test chunking multiple documents."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        documents = [
            {"content": "First document content.", "metadata": {"id": "doc1"}},
            {"content": "Second document with longer content that exceeds the chunk size limit.", "metadata": {"id": "doc2"}}
        ]
        
        chunks = chunker.chunk_documents(documents)
        
        assert len(chunks) > 0
        
        # Check document indexing
        doc1_chunks = [c for c in chunks if c["metadata"]["id"] == "doc1"]
        doc2_chunks = [c for c in chunks if c["metadata"]["id"] == "doc2"]
        
        assert len(doc1_chunks) >= 1
        assert len(doc2_chunks) >= 1
        
        # Check document index metadata
        for chunk in chunks:
            assert "document_index" in chunk["metadata"]
            assert chunk["metadata"]["document_index"] in [0, 1]

    def test_get_chunk_info(self):
        """Test getting chunker configuration info."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        
        info = chunker.get_chunk_info()
        
        assert info["chunk_size"] == 200
        assert info["chunk_overlap"] == 20
        assert info["effective_chunk_size"] == 180  # 200 - 20

    def test_find_best_break_point(self):
        """Test the break point finding logic."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "First sentence. Second sentence.\n\nNew paragraph here. More content."
        
        # Test paragraph break preference
        break_point = chunker._find_best_break_point(text, 0, 40)
        
        # Should find a reasonable break point
        assert 0 < break_point <= 40
        
        # Should prefer natural boundaries
        break_text = text[:break_point]
        assert any(delimiter in break_text for delimiter in ['. ', '\n\n', '\n'])

    def test_chunk_edge_cases(self):
        """Test edge cases in chunking."""
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        
        # Very short chunk size with long words
        text = "supercalifragilisticexpialidocious antidisestablishmentarianism"
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        # Should handle long words gracefully
        for chunk in chunks:
            assert len(chunk["content"]) > 0

    def test_chunk_text_with_only_whitespace_differences(self):
        """Test chunking text that differs only in whitespace."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text1 = "Text with normal spaces"
        text2 = "Text  with   extra    spaces"
        
        chunks1 = chunker.chunk_text(text1)
        chunks2 = chunker.chunk_text(text2)
        
        # Should handle both gracefully
        assert len(chunks1) > 0
        assert len(chunks2) > 0
