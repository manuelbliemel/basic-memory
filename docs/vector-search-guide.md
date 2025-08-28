# Vector Search Guide

## Overview

Basic Memory now includes semantic vector search capabilities that complement the existing FTS5 fuzzy search system. Vector search enables finding content based on semantic meaning rather than just exact word matches.

## Key Features

- **Multiple Search Strategies**: Choose between fuzzy-only, vector-only, or hybrid approaches
- **Local & API Embeddings**: Use local sentence-transformers models or OpenAI API
- **Transparent Integration**: Existing search APIs and MCP tools work unchanged
- **Auto-Sync**: Vector indices automatically update when files change
- **Configurable**: Customize embedding models, chunk sizes, and similarity thresholds

## Quick Start

### 1. Installation

Vector search dependencies are included in basic-memory. Install with:

```bash
pip install basic-memory
```

Optional dependencies are automatically available:
- ChromaDB for vector storage
- sentence-transformers for local embeddings  
- OpenAI client for API embeddings

### 2. Configuration

Configure vector search in `~/.basic-memory/config.json`:

```json
{
  "search_config": {
    "strategy": "hybrid",
    "vector": {
      "enabled": true,
      "provider": "chromadb",
      "embedding_model": "local",
      "embedding_model_name": "all-MiniLM-L6-v2",
      "chunk_size": 500,
      "chunk_overlap": 50,
      "similarity_threshold": 0.1,
      "max_results": 5
    }
  }
}
```

### 3. Search Strategies

Choose your preferred search strategy:

#### `fuzzy_only` (Default)
Uses only the existing FTS5 full-text search.
```json
{"search_config": {"strategy": "fuzzy_only"}}
```

#### `vector_only` 
Uses only semantic vector search.
```json
{"search_config": {"strategy": "vector_only"}}
```

#### `hybrid` (Recommended)
Combines both search types for best results.
```json
{"search_config": {"strategy": "hybrid"}}
```

#### `fuzzy_primary`
Uses fuzzy search first, falls back to vector if no results.
```json
{"search_config": {"strategy": "fuzzy_primary"}}
```

#### `vector_primary`
Uses vector search first, falls back to fuzzy if no results.
```json
{"search_config": {"strategy": "vector_primary"}}
```

## Embedding Models

### Local Embeddings (Offline)

Use local sentence-transformers models for privacy and offline operation:

```json
{
  "search_config": {
    "vector": {
      "embedding_model": "local",
      "embedding_model_name": "all-MiniLM-L6-v2"
    }
  }
}
```

**Recommended Models:**
- `all-MiniLM-L6-v2`: Fast, good quality (384 dimensions)
- `all-mpnet-base-v2`: Higher quality, slower (768 dimensions)
- `paraphrase-MiniLM-L6-v2`: Good for paraphrases
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A

### OpenAI API Embeddings

Use OpenAI's embedding API for high-quality embeddings:

```json
{
  "search_config": {
    "vector": {
      "embedding_model": "openai",
      "embedding_model_name": "text-embedding-ada-002",
      "api_key": "your-openai-api-key"
    }
  }
}
```

Set your API key via:
- Configuration file: `"api_key": "sk-..."`
- Environment variable: `OPENAI_API_KEY=sk-...`

**Supported Models:**
- `text-embedding-ada-002`: 1536 dimensions, $0.0001/1K tokens
- `text-embedding-3-small`: 1536 dimensions, $0.00002/1K tokens  
- `text-embedding-3-large`: 3072 dimensions, $0.00013/1K tokens

## Advanced Configuration

### Chunking Settings

Control how large documents are split for vector indexing:

```json
{
  "search_config": {
    "vector": {
      "chunk_size": 500,        // Max characters per chunk
      "chunk_overlap": 50,      // Characters to overlap between chunks
      "similarity_threshold": 0.3,  // Minimum similarity score (0.0-1.0)
      "max_results": 50         // Maximum results from vector search
    }
  }
}
```

### Provider Settings

```json
{
  "search_config": {
    "vector": {
      "provider": "chromadb",   // Vector database (currently only ChromaDB)
      "enabled": true           // Enable/disable vector search
    }
  }
}
```

## Usage Examples

### Basic Search (No Code Changes)

Vector search works transparently with existing tools:

```bash
# CLI search
basic-memory search "machine learning concepts"

# MCP search tool
# Works exactly the same as before
```

### API Usage

All existing API endpoints work unchanged:

```python
# POST /search
{
  "text": "artificial intelligence",
  "limit": 10
}
```

### Semantic vs Fuzzy Examples

**Fuzzy Search**: Finds exact word matches
- Query: "ML algorithm" → Finds documents containing "ML" and "algorithm"

**Vector Search**: Finds semantic matches  
- Query: "ML algorithm" → Finds documents about "machine learning", "neural networks", "classification methods"

**Hybrid Search**: Best of both
- Combines exact matches with semantic understanding
- Removes duplicates automatically

## How It Works

### Indexing Process

1. **File Changes Detected**: WatchService detects markdown file changes
2. **Dual Indexing**: SyncService updates both FTS5 and vector indices
3. **Content Chunking**: Large documents split into searchable chunks
4. **Embedding Generation**: Text converted to vectors using chosen model
5. **Vector Storage**: Embeddings stored in ChromaDB with metadata

### Search Process

1. **Strategy Selection**: Configuration determines search approach
2. **Query Processing**: Text queries generate embeddings for vector search
3. **Parallel Execution**: Fuzzy and vector searches run concurrently (hybrid mode)
4. **Result Combination**: Results merged and deduplicated by permalink
5. **Relevance Scoring**: Combined results sorted by relevance

## Troubleshooting

### Vector Search Not Working

1. **Check Configuration**: Ensure `vector.enabled: true`
2. **Verify Dependencies**: Ensure `chromadb` and embedding libraries installed
3. **Check Logs**: Look for initialization errors in logs
4. **Test Fallback**: Vector search failures automatically fall back to fuzzy search

### Performance Issues

1. **Adjust Chunk Size**: Smaller chunks = faster indexing, but may reduce context
2. **Change Embedding Model**: Local models are faster than API calls
3. **Tune Similarity Threshold**: Higher threshold = fewer but more relevant results

### API Key Issues (OpenAI)

1. **Set API Key**: Use config file or `OPENAI_API_KEY` environment variable
2. **Check Permissions**: Ensure API key has embeddings access
3. **Monitor Usage**: OpenAI charges per token for embeddings

## Storage and Data

### Vector Database Location

ChromaDB stores vector data in:
```
~/.basic-memory/chroma/
```

### Data Persistence

- Vector indices persist across restarts
- Automatic cleanup when entities are deleted
- Collections organized by project for isolation

### Backup Considerations

To backup vector search data:
1. Include `~/.basic-memory/chroma/` in backups
2. Vector indices can be rebuilt from markdown files if needed
3. Configuration in `~/.basic-memory/config.json`

## Migration and Upgrades

### Enabling Vector Search

1. Update configuration to enable vector search
2. Existing FTS5 search continues working immediately
3. Vector indexing happens automatically on next file changes
4. Force reindex: `basic-memory sync --reindex`

### Switching Embedding Models

1. Update configuration with new model
2. Clear vector index: Delete `~/.basic-memory/chroma/`
3. Reindex: `basic-memory sync --reindex`

### Disabling Vector Search

```json
{"search_config": {"vector": {"enabled": false}}}
```

Vector search components are disabled but FTS5 search continues unchanged.

## Performance Benchmarks

### Local Embeddings (all-MiniLM-L6-v2)
- **Initialization**: ~2-5 seconds (model download on first use)
- **Indexing**: ~10-50ms per document chunk
- **Search**: ~5-20ms per query
- **Memory**: ~200-500MB for model

### OpenAI Embeddings  
- **Initialization**: <1 second
- **Indexing**: ~100-500ms per API call (batch processing)
- **Search**: ~100-300ms per API call
- **Cost**: ~$0.0001 per 1K tokens (ada-002)

### Storage Requirements
- **ChromaDB**: ~1-5MB per 1000 documents
- **Embeddings**: ~1.5KB per chunk (384-dim) to ~12KB per chunk (3072-dim)
