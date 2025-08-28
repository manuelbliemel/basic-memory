"""Text chunking utilities for vector search indexing."""

from typing import List, Optional


class TextChunker:
    """Utility class for splitting large documents into searchable chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the text chunker.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        """Split text into chunks with optional metadata.
        
        Args:
            text: Text to be chunked
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with 'content' and 'metadata' keys
        """
        if not text or not text.strip():
            return []

        # Clean and normalize the text
        clean_text = text.strip()
        
        # If text is shorter than chunk size, return as single chunk
        if len(clean_text) <= self.chunk_size:
            chunk_metadata = dict(metadata) if metadata else {}
            chunk_metadata.update({
                'chunk_index': 0,
                'total_chunks': 1,
                'chunk_length': len(clean_text)
            })
            return [{
                'content': clean_text,
                'metadata': chunk_metadata
            }]

        chunks = []
        start_pos = 0
        chunk_index = 0
        
        while start_pos < len(clean_text):
            # Calculate end position for this chunk
            end_pos = min(start_pos + self.chunk_size, len(clean_text))
            
            # Try to break at a natural boundary (sentence, paragraph, or word)
            if end_pos < len(clean_text):
                end_pos = self._find_best_break_point(clean_text, start_pos, end_pos)
            
            # Extract chunk content
            chunk_content = clean_text[start_pos:end_pos].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunk_metadata = dict(metadata) if metadata else {}
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'chunk_start': start_pos,
                    'chunk_end': end_pos,
                    'chunk_length': len(chunk_content)
                })
                
                chunks.append({
                    'content': chunk_content,
                    'metadata': chunk_metadata
                })
                chunk_index += 1
            
            # Move start position forward, accounting for overlap
            if end_pos >= len(clean_text):
                break
                
            next_start = end_pos - self.chunk_overlap
            start_pos = max(next_start, start_pos + 1)  # Ensure progress

        # Update total chunks count in all metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = total_chunks

        return chunks

    def _find_best_break_point(self, text: str, start_pos: int, target_end: int) -> int:
        """Find the best position to break text for natural chunking.
        
        Prioritizes breaking at:
        1. Markdown headings (# ## ### etc.)
        2. Double newlines (paragraph breaks)
        3. Single newlines
        4. Sentence endings (. ! ?)
        5. Word boundaries (spaces)
        6. Target position as fallback
        """
        # Look for natural breaks, starting with the most semantic
        search_start = max(start_pos, target_end - 100)  # Don't search too far back
        
        # First priority: Markdown headings (represent topic/chapter changes)
        import re
        heading_pattern = r'\n(#{1,6}\s)'  # Matches \n# \n## \n### etc.
        for match in re.finditer(heading_pattern, text[search_start:target_end]):
            heading_pos = search_start + match.start() + 1  # +1 to skip the \n
            if heading_pos > start_pos:  # Ensure we make progress
                return heading_pos
        
        # Second priority: Other natural breaks
        for break_char in ['\n\n', '\n', '. ', '! ', '? ']:
            break_pos = text.rfind(break_char, search_start, target_end)
            if break_pos != -1:
                return break_pos + len(break_char)
        
        # Fall back to word boundary
        space_pos = text.rfind(' ', search_start, target_end)
        if space_pos != -1:
            return space_pos + 1
            
        # Last resort: break at target position
        return target_end

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        """Chunk multiple documents efficiently.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata' keys
            
        Returns:
            List of chunks from all documents with preserved metadata
        """
        all_chunks = []
        
        for doc_index, document in enumerate(documents):
            content = document.get('content', '')
            base_metadata = document.get('metadata', {})
            
            # Add document-level metadata
            base_metadata['document_index'] = doc_index
            
            # Chunk this document
            doc_chunks = self.chunk_text(content, base_metadata)
            all_chunks.extend(doc_chunks)
            
        return all_chunks

    def get_chunk_info(self) -> dict:
        """Get information about chunker configuration.
        
        Returns:
            Dictionary with chunker settings
        """
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'effective_chunk_size': self.chunk_size - self.chunk_overlap
        }
