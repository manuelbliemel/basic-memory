"""Tests for search result combination functionality."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from basic_memory.vector.combiner import SearchResultCombiner
from basic_memory.repository.search_repository import SearchIndexRow


class TestSearchResultCombiner:
    """Test suite for SearchResultCombiner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.combiner = SearchResultCombiner()
        
        # Mock fuzzy search results
        self.fuzzy_results = [
            SearchIndexRow(
                id=1, project_id=1, type="entity", file_path="/test1.md",
                title="Test Entity 1", permalink="test-entity-1",
                created_at=datetime.now(), updated_at=datetime.now(),
                content_snippet="This is test content 1", score=0.8
            ),
            SearchIndexRow(
                id=2, project_id=1, type="entity", file_path="/test2.md", 
                title="Test Entity 2", permalink="test-entity-2",
                created_at=datetime.now(), updated_at=datetime.now(),
                content_snippet="This is test content 2", score=0.6
            )
        ]
        
        # Mock vector search results
        self.vector_results = [
            {
                "id": "test-entity-1",
                "content": "This is test content 1 with more detail",
                "metadata": {
                    "entity_id": "1",
                    "project_id": "1",
                    "type": "entity",
                    "title": "Test Entity 1",
                    "permalink": "test-entity-1",
                    "file_path": "/test1.md"
                },
                "score": 0.5
            },
            {
                "id": "test-entity-3",
                "content": "This is test content 3",
                "metadata": {
                    "entity_id": "3",
                    "project_id": "1", 
                    "type": "entity",
                    "title": "Test Entity 3",
                    "permalink": "test-entity-3",
                    "file_path": "/test3.md"
                },
                "score": 0.3
            }
        ]

    def test_combine_results_hybrid(self):
        """Test combining results with hybrid strategy (simplified interface)."""
        results = self.combiner.combine_results(
            fuzzy_results=self.fuzzy_results,
            vector_results=self.vector_results
        )
        
        # Should have 3 unique results (2 fuzzy + 1 unique vector)
        # test-entity-1 appears in both so should be deduplicated
        assert len(results) == 3
        
        permalinks = [r.permalink for r in results]
        assert "test-entity-1" in permalinks
        assert "test-entity-2" in permalinks  
        assert "test-entity-3" in permalinks

    def test_combine_results_empty_fuzzy(self):
        """Test combining results when fuzzy results are empty."""
        results = self.combiner.combine_results(
            fuzzy_results=[],
            vector_results=self.vector_results
        )
        
        assert len(results) == 2
        assert all(isinstance(r, SearchIndexRow) for r in results)
        assert results[0].title == "Test Entity 1"
        assert results[1].title == "Test Entity 3"

    def test_combine_results_empty_vector(self):
        """Test combining results when vector results are empty."""
        results = self.combiner.combine_results(
            fuzzy_results=self.fuzzy_results,
            vector_results=[]
        )
        
        assert len(results) == 2
        assert results == self.fuzzy_results

    def test_combine_results_both_empty(self):
        """Test combining results when both are empty."""
        results = self.combiner.combine_results(
            fuzzy_results=[],
            vector_results=[]
        )
        
        assert len(results) == 0

    def test_convert_vector_to_search_rows(self):
        """Test converting vector results to SearchIndexRow format."""
        results = self.combiner._convert_vector_to_search_rows(self.vector_results)
        
        assert len(results) == 2
        assert all(isinstance(r, SearchIndexRow) for r in results)
        
        # Check first result
        assert results[0].title == "Test Entity 1"
        assert results[0].permalink == "test-entity-1"
        assert results[0].entity_id == 1  # Converted from string
        assert results[0].project_id == 1
        assert results[0].score == 0.5

    def test_safe_int_convert(self):
        """Test safe integer conversion."""
        # Valid conversions
        assert self.combiner._safe_int_convert("123") == 123
        assert self.combiner._safe_int_convert(456) == 456
        assert self.combiner._safe_int_convert("0") == 0
        
        # Invalid conversions
        assert self.combiner._safe_int_convert(None) is None
        assert self.combiner._safe_int_convert("invalid") is None
        assert self.combiner._safe_int_convert("") is None
        assert self.combiner._safe_int_convert([1, 2, 3]) is None

    def test_merge_hybrid_deduplication(self):
        """Test that hybrid merge properly deduplicates by permalink."""
        # Create overlapping results
        fuzzy = [
            SearchIndexRow(
                id=1, project_id=1, type="entity", file_path="/test.md",
                title="Test", permalink="same-permalink",
                created_at=datetime.now(), updated_at=datetime.now()
            )
        ]
        
        vector = [
            SearchIndexRow(
                id=2, project_id=1, type="entity", file_path="/test.md", 
                title="Test Vector", permalink="same-permalink",
                created_at=datetime.now(), updated_at=datetime.now()
            )
        ]
        
        results = self.combiner._merge_hybrid(fuzzy, vector)
        
        # Should have only one result (deduplicated)
        assert len(results) == 1
        assert results[0].title == "Test"  # Fuzzy takes precedence

    def test_sort_by_relevance(self):
        """Test relevance-based sorting with score type awareness."""
        results = [
            SearchIndexRow(
                id=1, project_id=1, type="observation", file_path="/test.md",
                title="BM25 Best", permalink="bm25-best", score=-21.0,  # BM25: more negative = better
                created_at=datetime.now(), updated_at=datetime.now(),
                metadata={"score_type": "fuzzy_bm25"}
            ),
            SearchIndexRow(
                id=2, project_id=1, type="entity", file_path="/test.md", 
                title="Vector High Score", permalink="vector-high", score=0.9,  # Similarity: higher = better
                created_at=datetime.now(), updated_at=datetime.now(),
                metadata={"score_type": "vector_similarity"}
            ),
            SearchIndexRow(
                id=3, project_id=1, type="entity", file_path="/test.md",
                title="BM25 Worst", permalink="bm25-worst", score=-9.0,  # BM25: less negative = worse
                created_at=datetime.now(), updated_at=datetime.now(),
                metadata={"score_type": "fuzzy_bm25"}
            ),
            SearchIndexRow(
                id=4, project_id=1, type="relation", file_path="/test.md",
                title="No Score", permalink="none",
                created_at=datetime.now(), updated_at=datetime.now()
            )
        ]
        
        sorted_results = self.combiner.sort_by_relevance(results)
        
        # After normalization with correct BM25 handling:
        # - BM25 Best (-21): abs(-21)/(abs(-21)+1) = 21/22 = 0.954
        # - Vector High Score (0.9): 0.9 (unchanged)  
        # - BM25 Worst (-9): abs(-9)/(abs(-9)+1) = 9/10 = 0.9
        # - No Score: 0.3 + 0.2 = 0.5 (entity type bonus)
        
        assert sorted_results[0].title == "BM25 Best"        # 0.954
        assert sorted_results[1].title == "Vector High Score" or sorted_results[1].title == "BM25 Worst"  # 0.9 (tie)
        assert sorted_results[3].title == "No Score"        # 0.5

    def test_limit_results(self):
        """Test result limiting."""
        results = self.fuzzy_results
        
        # Limit to 1
        limited = self.combiner.limit_results(results, 1)
        assert len(limited) == 1
        
        # Limit larger than available
        limited = self.combiner.limit_results(results, 10)
        assert len(limited) == 2
        
        # Zero or negative limit
        limited = self.combiner.limit_results(results, 0)
        assert len(limited) == 2
        
        limited = self.combiner.limit_results(results, -1)
        assert len(limited) == 2

    def test_apply_offset(self):
        """Test offset application."""
        results = self.fuzzy_results
        
        # Apply offset 1
        offset_results = self.combiner.apply_offset(results, 1)
        assert len(offset_results) == 1
        assert offset_results[0] == results[1]
        
        # Offset larger than available
        offset_results = self.combiner.apply_offset(results, 10)
        assert len(offset_results) == 0
        
        # Zero or negative offset
        offset_results = self.combiner.apply_offset(results, 0)
        assert len(offset_results) == 2
        
        offset_results = self.combiner.apply_offset(results, -1)
        assert len(offset_results) == 2

    def test_convert_vector_results_with_missing_metadata(self):
        """Test converting vector results with incomplete metadata."""
        incomplete_vector_results = [
            {
                "id": "test-1",
                "content": "Test content",
                "metadata": {
                    "entity_id": "1",
                    # Missing some fields
                },
                "score": 0.8
            }
        ]
        
        results = self.combiner._convert_vector_to_search_rows(incomplete_vector_results)
        
        assert len(results) == 1
        assert results[0].title == ""  # Default empty string
        assert results[0].entity_id == 1
        assert results[0].project_id == 1  # Default value

    def test_convert_vector_results_with_invalid_data(self):
        """Test converting vector results with invalid data that causes exceptions."""
        # This should be handled gracefully
        invalid_vector_results = [
            {
                "id": "test-1", 
                "content": "Test content",
                "metadata": {
                    "entity_id": "not-a-number",  # Invalid integer
                },
                "score": 0.8
            }
        ]
        
        # Should not raise exception, but may skip invalid entries
        results = self.combiner._convert_vector_to_search_rows(invalid_vector_results)
        
        # Should handle gracefully (either convert or skip)
        assert isinstance(results, list)
