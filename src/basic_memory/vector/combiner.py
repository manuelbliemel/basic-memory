"""Result combination and deduplication logic for search results."""

from typing import List, Dict, Any, Set, Optional
from datetime import datetime
from loguru import logger

from basic_memory.repository.search_repository import SearchIndexRow


class SearchResultCombiner:
    """Combines and deduplicates search results from multiple sources."""

    def __init__(self):
        pass

    def combine_results(
        self,
        fuzzy_results: List[SearchIndexRow],
        vector_results: List[Dict[str, Any]]
    ) -> List[SearchIndexRow]:
        """Combine fuzzy and vector search results with hybrid deduplication.
        
        Args:
            fuzzy_results: Results from FTS5 fuzzy search
            vector_results: Results from vector search
            
        Returns:
            Combined and deduplicated list of SearchIndexRow objects
        """
        # Convert vector results to SearchIndexRow format
        converted_vector_results = self._convert_vector_to_search_rows(vector_results)
        
        # Merge using hybrid strategy (union with deduplication)
        return self._merge_hybrid(fuzzy_results, converted_vector_results)

    def _convert_vector_to_search_rows(self, vector_results: List[Dict[str, Any]]) -> List[SearchIndexRow]:
        """Convert vector search results to SearchIndexRow format."""
        search_rows = []
        
        for result in vector_results:
            try:
                metadata = result.get('metadata', {})
                
                # Extract values from metadata, handling string conversion
                entity_id = self._safe_int_convert(metadata.get('entity_id'))
                project_id = self._safe_int_convert(metadata.get('project_id'))
                from_id = self._safe_int_convert(metadata.get('from_id'))
                to_id = self._safe_int_convert(metadata.get('to_id'))
                
                # Extract timestamps from metadata
                created_at = metadata.get('created_at')
                updated_at = metadata.get('updated_at')
                
                # Add score type to metadata for proper handling
                enhanced_metadata = dict(metadata)
                enhanced_metadata['score_type'] = result.get('score_type', 'vector_similarity')
                
                # Create SearchIndexRow from vector result
                search_row = SearchIndexRow(
                    id=entity_id if entity_id else 0,  # Use entity_id as primary ID
                    type=metadata.get('type', 'entity'),
                    title=metadata.get('title', ''),
                    content_stems=result.get('content', ''),
                    content_snippet=result.get('content', '')[:250] + '...' if len(result.get('content', '')) > 250 else result.get('content', ''),
                    permalink=metadata.get('permalink', ''),
                    file_path=metadata.get('file_path', ''),
                    category=metadata.get('category'),
                    entity_id=entity_id,
                    from_id=from_id,
                    to_id=to_id,
                    relation_type=metadata.get('relation_type'),
                    metadata=enhanced_metadata,  # Include score type
                    created_at=created_at or datetime.now(),  # Use actual timestamps or fallback to current
                    updated_at=updated_at or datetime.now(),
                    project_id=project_id if project_id is not None else 1,  # Default to project 1 if not available
                    score=result.get('score', 0.0)  # Include vector similarity score
                )
                search_rows.append(search_row)
                
            except Exception as e:
                logger.warning(f"Failed to convert vector result to SearchIndexRow: {e}")
                continue
                
        return search_rows

    def _safe_int_convert(self, value: Any) -> Optional[int]:
        """Safely convert a value to integer, handling string conversions."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _merge_hybrid(
        self,
        fuzzy_results: List[SearchIndexRow],
        vector_results: List[SearchIndexRow]
    ) -> List[SearchIndexRow]:
        """Merge results using hybrid strategy (union with deduplication)."""
        # Use permalink as deduplication key since it's unique per item
        seen_permalinks: Set[str] = set()
        combined_results = []
        
        # Add fuzzy results first (they may have better metadata completeness)
        for result in fuzzy_results:
            if result.permalink and result.permalink not in seen_permalinks:
                seen_permalinks.add(result.permalink)
                combined_results.append(result)
            elif not result.permalink:
                # Handle items without permalinks (shouldn't happen but be safe)
                combined_results.append(result)
        
        # Add vector results that don't duplicate fuzzy results
        for result in vector_results:
            if result.permalink and result.permalink not in seen_permalinks:
                seen_permalinks.add(result.permalink)
                combined_results.append(result)
            elif not result.permalink:
                # Handle items without permalinks
                combined_results.append(result)
        
        logger.debug(f"Hybrid merge: {len(fuzzy_results)} fuzzy + {len(vector_results)} vector = {len(combined_results)} combined")
        return combined_results

    def sort_by_relevance(
        self,
        results: List[SearchIndexRow]
    ) -> List[SearchIndexRow]:
        """Sort results by normalized relevance score.
        
        Handles score compatibility between different search types:
        - Fuzzy: BM25 scores (lower = better, like distance)  
        - Vector: Similarity scores (higher = better)
        """
        def normalized_relevance_score(result: SearchIndexRow) -> float:
            if hasattr(result, 'score') and result.score is not None:
                # Check if this is a vector result with similarity score
                if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                    score_type = result.metadata.get('score_type', 'fuzzy_bm25')
                    
                    if score_type == 'vector_similarity':
                        # Vector similarity: higher = better (already normalized 0-1)
                        return result.score
                    else:
                        # Fuzzy BM25: negative scores where more negative = better
                        # Convert to 0-1 range where higher = better  
                        # Examples: -21 (best) → 0.954, -9 (worse) → 0.9
                        return abs(result.score) / (abs(result.score) + 1)
                else:
                    # Assume fuzzy BM25 if no metadata
                    return 1.0 / (1.0 + result.score)
            
            # Fall back to type-based heuristics for results without scores
            base_score = 0.3
            
            # Type preferences (can be adjusted based on use case)
            if result.type == 'entity':
                base_score += 0.2
            elif result.type == 'observation':
                base_score += 0.1
            elif result.type == 'relation':
                base_score += 0.05
                
            # Content length bonus
            if result.content_snippet and len(result.content_snippet) > 100:
                base_score += 0.05
                
            return base_score
        
        return sorted(results, key=normalized_relevance_score, reverse=True)

    def limit_results(self, results: List[SearchIndexRow], limit: int) -> List[SearchIndexRow]:
        """Apply limit to results while preserving order."""
        if limit <= 0:
            return results
        return results[:limit]

    def apply_offset(self, results: List[SearchIndexRow], offset: int) -> List[SearchIndexRow]:
        """Apply offset to results for pagination."""
        if offset <= 0:
            return results
        return results[offset:]
