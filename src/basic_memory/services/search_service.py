"""Service for search operations."""

import ast
from datetime import datetime
from typing import List, Optional, Set

from dateparser import parse
from fastapi import BackgroundTasks
from loguru import logger
from sqlalchemy import text

from basic_memory.models import Entity
from basic_memory.repository import EntityRepository
from basic_memory.repository.search_repository import SearchRepository, SearchIndexRow
from basic_memory.schemas.search import SearchQuery, SearchItemType
from basic_memory.services import FileService
from basic_memory.vector.base import VectorSearchProvider, EmbeddingProvider
from basic_memory.vector.combiner import SearchResultCombiner
from basic_memory.vector.chunker import TextChunker
from basic_memory.vector.config import SearchStrategy, EmbeddingModelType
from basic_memory.vector.providers.chromadb_provider import ChromaDBProvider
from basic_memory.vector.embeddings.local_embedding import LocalEmbeddingProvider
from basic_memory.vector.embeddings.openai_embedding import OpenAIEmbeddingProvider
from basic_memory.config import ConfigManager


class SearchService:
    """Service for search operations.

    Supports three primary search modes:
    1. Exact permalink lookup
    2. Pattern matching with * (e.g., 'specs/*')
    3. Full-text search across title/content
    """

    def __init__(
        self,
        search_repository: SearchRepository,
        entity_repository: EntityRepository,
        file_service: FileService,
    ):
        self.repository = search_repository
        self.entity_repository = entity_repository
        self.file_service = file_service
        
        # Vector search components (initialized lazily)
        self.vector_provider: Optional[VectorSearchProvider] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.text_chunker: Optional[TextChunker] = None
        self.result_combiner: SearchResultCombiner = SearchResultCombiner()
        self._vector_initialized = False

    async def init_search_index(self):
        """Create FTS5 virtual table if it doesn't exist."""
        await self.repository.init_search_index()

    async def init_vector_search(self) -> None:
        """Initialize vector search provider and embeddings."""
        if self._vector_initialized:
            return

        try:
            # Get configuration
            config_manager = ConfigManager()
            config = config_manager.load_config()
            search_config = config.search_config
            
            if not search_config.vector.enabled:
                logger.info("Vector search is disabled in configuration")
                return

            logger.info("Initializing vector search...")

            # Initialize embedding provider
            if search_config.vector.embedding_model == EmbeddingModelType.LOCAL:
                self.embedding_provider = LocalEmbeddingProvider(
                    model_name=search_config.vector.embedding_model_name
                )
            elif search_config.vector.embedding_model == EmbeddingModelType.OPENAI:
                self.embedding_provider = OpenAIEmbeddingProvider(
                    model_name=search_config.vector.embedding_model_name,
                    api_key=search_config.vector.api_key
                )
            else:
                logger.warning(f"Unsupported embedding model type: {search_config.vector.embedding_model}")
                return

            # Initialize embedding provider
            await self.embedding_provider.initialize()

            # Initialize vector provider
            if search_config.vector.provider == "chromadb":
                self.vector_provider = ChromaDBProvider(
                    embedding_provider=self.embedding_provider
                )
                await self.vector_provider.initialize()
            else:
                logger.warning(f"Unsupported vector provider: {search_config.vector.provider}")
                return

            # Initialize text chunker
            self.text_chunker = TextChunker(
                chunk_size=search_config.vector.chunk_size,
                chunk_overlap=search_config.vector.chunk_overlap
            )

            self._vector_initialized = True
            logger.info("Vector search initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            # Don't fail the entire service if vector search fails
            self._vector_initialized = False

    async def reindex_all(self, background_tasks: Optional[BackgroundTasks] = None) -> None:
        """Reindex all content from database."""

        logger.info("Starting full reindex")
        # Clear and recreate search index
        await self.repository.execute_query(text("DROP TABLE IF EXISTS search_index"), params={})
        await self.init_search_index()

        # Reindex all entities
        logger.debug("Indexing entities")
        entities = await self.entity_repository.find_all()
        for entity in entities:
            await self.index_entity(entity, background_tasks)

        logger.info("Reindex complete")

    async def search(self, query: SearchQuery, limit=10, offset=0) -> List[SearchIndexRow]:
        """Search across all indexed content with configurable search strategy.

        Supports three modes:
        1. Exact permalink: finds direct matches for a specific path
        2. Pattern match: handles * wildcards in paths
        3. Text search: full-text search across title/content

        Supports multiple search strategies:
        - fuzzy_only: Use only FTS5 search
        - vector_only: Use only vector search  
        - fuzzy_primary: Fuzzy first, fallback to vector
        - vector_primary: Vector first, fallback to fuzzy
        - hybrid: Both searches, union results
        """
        if query.no_criteria():
            logger.debug("no criteria passed to query")
            return []

        logger.trace(f"Searching with query: {query}")

        # Get search strategy from configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        strategy = config.search_config.strategy

        # Handle non-text searches (permalink, pattern matching) with fuzzy search only
        if query.permalink or query.permalink_match or not query.text:
            return await self._fuzzy_search(query, limit, offset)

        # Initialize vector search if needed and enabled
        if strategy != SearchStrategy.FUZZY_ONLY:
            await self.init_vector_search()

        # Execute search based on strategy
        if strategy == SearchStrategy.FUZZY_ONLY or not self._vector_initialized:
            return await self._fuzzy_search(query, limit, offset)
        elif strategy == SearchStrategy.VECTOR_ONLY:
            return await self._vector_search(query, limit, offset)
        elif strategy == SearchStrategy.FUZZY_PRIMARY:
            return await self._fuzzy_primary_search(query, limit, offset)
        elif strategy == SearchStrategy.VECTOR_PRIMARY:
            return await self._vector_primary_search(query, limit, offset)
        else:  # hybrid
            return await self._hybrid_search(query, limit, offset)

    async def _fuzzy_search(self, query: SearchQuery, limit: int, offset: int) -> List[SearchIndexRow]:
        """Execute FTS5 fuzzy search."""
        after_date = (
            (
                query.after_date
                if isinstance(query.after_date, datetime)
                else parse(query.after_date)
            )
            if query.after_date
            else None
        )

        results = await self.repository.search(
            search_text=query.text,
            permalink=query.permalink,
            permalink_match=query.permalink_match,
            title=query.title,
            types=query.types,
            search_item_types=query.entity_types,
            after_date=after_date,
            limit=limit,
            offset=offset,
        )

        logger.debug(f"Fuzzy search returned {len(results)} results")
        return results

    async def _vector_search(self, query: SearchQuery, limit: int, offset: int) -> List[SearchIndexRow]:
        """Execute vector similarity search."""
        if not self._vector_initialized or not self.vector_provider or not self.embedding_provider:
            logger.warning("Vector search not initialized, falling back to fuzzy search")
            return await self._fuzzy_search(query, limit, offset)

        if not query.text:
            logger.debug("No text query provided for vector search")
            return []

        try:
            # Generate query embeddings
            query_embeddings = await self.embedding_provider.embed_text(query.text)
            
            # Get configuration for search parameters
            config_manager = ConfigManager()
            config = config_manager.load_config()
            vector_config = config.search_config.vector

            # Build metadata filters to match FTS5 repository filters
            metadata_filter = {}
            
            # Filter by entity types (maps to 'types' parameter)
            if query.types:
                metadata_filter["entity_type"] = {"$in": query.types}
            
            # Filter by search item types (entity, observation, relation)
            if query.entity_types:
                item_type_values = [t.value for t in query.entity_types]
                metadata_filter["type"] = {"$in": item_type_values}
            
            # Title filtering (if needed for vector search)
            if query.title:
                metadata_filter["title"] = {"$regex": f".*{query.title}.*"}
            
            # Project ID filtering for project-scoped search
            if self.repository.project_id:
                metadata_filter["project_id"] = str(self.repository.project_id)

            # Execute vector search with filters
            vector_results = await self.vector_provider.search(
                query_embeddings=query_embeddings,
                limit=limit + offset,  # Get extra results to handle offset
                threshold=vector_config.similarity_threshold,
                metadata_filter=metadata_filter if metadata_filter else None
            )

            # Convert to SearchIndexRow format
            converted_results = self.result_combiner._convert_vector_to_search_rows(vector_results)
            
            # Apply date filtering (post-processing since ChromaDB doesn't handle datetime well)
            if query.after_date:
                after_date = (
                    query.after_date if isinstance(query.after_date, datetime) 
                    else parse(query.after_date)
                )
                converted_results = [
                    result for result in converted_results 
                    if result.created_at is not None and after_date is not None and result.created_at > after_date
                ]
            
            # Apply pagination
            paginated_results = self.result_combiner.apply_offset(converted_results, offset)
            final_results = self.result_combiner.limit_results(paginated_results, limit)

            logger.debug(f"Vector search returned {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}, falling back to fuzzy search")
            return await self._fuzzy_search(query, limit, offset)

    async def _fuzzy_primary_search(self, query: SearchQuery, limit: int, offset: int) -> List[SearchIndexRow]:
        """Execute fuzzy search first, fallback to vector if no results."""
        fuzzy_results = await self._fuzzy_search(query, limit, offset)
        if fuzzy_results:
            return fuzzy_results
        
        logger.debug("Fuzzy search returned no results, trying vector search")
        return await self._vector_search(query, limit, offset)

    async def _vector_primary_search(self, query: SearchQuery, limit: int, offset: int) -> List[SearchIndexRow]:
        """Execute vector search first, fallback to fuzzy if no results."""
        vector_results = await self._vector_search(query, limit, offset)
        if vector_results:
            return vector_results
        
        logger.debug("Vector search returned no results, trying fuzzy search")
        return await self._fuzzy_search(query, limit, offset)

    async def _hybrid_search(self, query: SearchQuery, limit: int, offset: int) -> List[SearchIndexRow]:
        """Execute both searches and combine results."""
        try:
            # Execute both searches concurrently - get enough results to handle offset
            fuzzy_results = await self._fuzzy_search(query, limit + offset, 0)
            vector_results_raw = []

            if self._vector_initialized and self.vector_provider and self.embedding_provider and query.text:
                # Generate query embeddings
                query_embeddings = await self.embedding_provider.embed_text(query.text)
                
                # Get configuration for search parameters
                config_manager = ConfigManager()
                config = config_manager.load_config()
                vector_config = config.search_config.vector

                # Build metadata filters to match FTS5 repository filters
                metadata_filter = {}
                
                # Filter by entity types (maps to 'types' parameter)
                if query.types:
                    metadata_filter["entity_type"] = {"$in": query.types}
                
                # Filter by search item types (entity, observation, relation)
                if query.entity_types:
                    item_type_values = [t.value for t in query.entity_types]
                    metadata_filter["type"] = {"$in": item_type_values}
                
                # Title filtering (if needed for vector search)
                if query.title:
                    metadata_filter["title"] = {"$regex": f".*{query.title}.*"}
                
                # Project ID filtering for project-scoped search
                if self.repository.project_id:
                    metadata_filter["project_id"] = str(self.repository.project_id)

                # Execute vector search with filters
                vector_results_raw = await self.vector_provider.search(
                    query_embeddings=query_embeddings,
                    limit=limit + offset,  # Get enough results to handle offset
                    threshold=vector_config.similarity_threshold,
                    metadata_filter=metadata_filter if metadata_filter else None
                )

            # Combine results
            combined_results = self.result_combiner.combine_results(
                fuzzy_results=fuzzy_results,
                vector_results=vector_results_raw
            )

            # Apply date filtering to combined results (post-processing since ChromaDB doesn't handle datetime well)
            if query.after_date:
                after_date = (
                    query.after_date if isinstance(query.after_date, datetime) 
                    else parse(query.after_date)
                )
                combined_results = [
                    result for result in combined_results 
                    if result.created_at is not None and after_date is not None and result.created_at > after_date
                ]

            # Apply pagination and sorting
            sorted_results = self.result_combiner.sort_by_relevance(combined_results)
            paginated_results = self.result_combiner.apply_offset(sorted_results, offset)
            final_results = self.result_combiner.limit_results(paginated_results, limit)

            logger.debug(f"Hybrid search returned {len(final_results)} combined results")
            return final_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}, falling back to fuzzy search")
            return await self._fuzzy_search(query, limit, offset)

    @staticmethod
    def _generate_variants(text: str) -> Set[str]:
        """Generate text variants for better fuzzy matching.

        Creates variations of the text to improve match chances:
        - Original form
        - Lowercase form
        - Path segments (for permalinks)
        - Common word boundaries
        """
        variants = {text, text.lower()}

        # Add path segments
        if "/" in text:
            variants.update(p.strip() for p in text.split("/") if p.strip())

        # Add word boundaries
        variants.update(w.strip() for w in text.lower().split() if w.strip())

        # Add trigrams for fuzzy matching
        variants.update(text[i : i + 3].lower() for i in range(len(text) - 2))

        return variants

    def _extract_entity_tags(self, entity: Entity) -> List[str]:
        """Extract tags from entity metadata for search indexing.

        Handles multiple tag formats:
        - List format: ["tag1", "tag2"]
        - String format: "['tag1', 'tag2']" or "[tag1, tag2]"
        - Empty: [] or "[]"

        Returns a list of tag strings for search indexing.
        """
        if not entity.entity_metadata or "tags" not in entity.entity_metadata:
            return []

        tags = entity.entity_metadata["tags"]

        # Handle list format (preferred)
        if isinstance(tags, list):
            return [str(tag) for tag in tags if tag]

        # Handle string format (legacy)
        if isinstance(tags, str):
            try:
                # Parse string representation of list
                parsed_tags = ast.literal_eval(tags)
                if isinstance(parsed_tags, list):
                    return [str(tag) for tag in parsed_tags if tag]
            except (ValueError, SyntaxError):
                # If parsing fails, treat as single tag
                return [tags] if tags.strip() else []

        return []  # pragma: no cover

    async def index_entity(
        self,
        entity: Entity,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> None:
        if background_tasks:
            background_tasks.add_task(self.index_entity_data, entity)
        else:
            await self.index_entity_data(entity)

    async def index_entity_data(
        self,
        entity: Entity,
    ) -> None:
        # delete all search index data associated with entity
        await self.repository.delete_by_entity_id(entity_id=entity.id)
        
        # Also delete from vector index if initialized
        if self._vector_initialized and self.vector_provider:
            try:
                await self.delete_entity_vector(entity.id)
            except Exception as e:
                logger.warning(f"Failed to delete vector index for entity {entity.id}: {e}")

        # reindex both FTS5 and vector
        await self.index_entity_markdown(
            entity
        ) if entity.is_markdown else await self.index_entity_file(entity)

    async def index_entity_file(
        self,
        entity: Entity,
    ) -> None:
        # Index entity file with no content
        await self.repository.index_item(
            SearchIndexRow(
                id=entity.id,
                entity_id=entity.id,
                type=SearchItemType.ENTITY.value,
                title=entity.title,
                file_path=entity.file_path,
                metadata={
                    "entity_type": entity.entity_type,
                },
                created_at=entity.created_at,
                updated_at=entity.updated_at,
                project_id=entity.project_id,
            )
        )

        # Also index in vector database if initialized
        if self._vector_initialized:
            try:
                await self.index_entity_vector(entity)
            except Exception as e:
                logger.warning(f"Failed to index entity {entity.id} in vector database: {e}")

    async def index_entity_markdown(
        self,
        entity: Entity,
    ) -> None:
        """Index an entity and all its observations and relations.

        Indexing structure:
        1. Entities
           - permalink: direct from entity (e.g., "specs/search")
           - file_path: physical file location
           - project_id: project context for isolation

        2. Observations
           - permalink: entity permalink + /observations/id (e.g., "specs/search/observations/123")
           - file_path: parent entity's file (where observation is defined)
           - project_id: inherited from parent entity

        3. Relations (only index outgoing relations defined in this file)
           - permalink: from_entity/relation_type/to_entity (e.g., "specs/search/implements/features/search-ui")
           - file_path: source entity's file (where relation is defined)
           - project_id: inherited from source entity

        Each type gets its own row in the search index with appropriate metadata.
        The project_id is automatically added by the repository when indexing.
        """

        content_stems = []
        content_snippet = ""
        title_variants = self._generate_variants(entity.title)
        content_stems.extend(title_variants)

        content = await self.file_service.read_entity_content(entity)
        if content:
            content_stems.append(content)
            content_snippet = f"{content[:250]}"

        if entity.permalink:
            content_stems.extend(self._generate_variants(entity.permalink))

        content_stems.extend(self._generate_variants(entity.file_path))

        # Add entity tags from frontmatter to search content
        entity_tags = self._extract_entity_tags(entity)
        if entity_tags:
            content_stems.extend(entity_tags)

        entity_content_stems = "\n".join(p for p in content_stems if p and p.strip())

        # Index entity
        await self.repository.index_item(
            SearchIndexRow(
                id=entity.id,
                type=SearchItemType.ENTITY.value,
                title=entity.title,
                content_stems=entity_content_stems,
                content_snippet=content_snippet,
                permalink=entity.permalink,
                file_path=entity.file_path,
                entity_id=entity.id,
                metadata={
                    "entity_type": entity.entity_type,
                },
                created_at=entity.created_at,
                updated_at=entity.updated_at,
                project_id=entity.project_id,
            )
        )

        # Index each observation with permalink
        for obs in entity.observations:
            # Index with parent entity's file path since that's where it's defined
            obs_content_stems = "\n".join(
                p for p in self._generate_variants(obs.content) if p and p.strip()
            )
            await self.repository.index_item(
                SearchIndexRow(
                    id=obs.id,
                    type=SearchItemType.OBSERVATION.value,
                    title=f"{obs.category}: {obs.content[:100]}...",
                    content_stems=obs_content_stems,
                    content_snippet=obs.content,
                    permalink=obs.permalink,
                    file_path=entity.file_path,
                    category=obs.category,
                    entity_id=entity.id,
                    metadata={
                        "tags": obs.tags,
                    },
                    created_at=entity.created_at,
                    updated_at=entity.updated_at,
                    project_id=entity.project_id,
                )
            )

        # Only index outgoing relations (ones defined in this file)
        for rel in entity.outgoing_relations:
            # Create descriptive title showing the relationship
            relation_title = (
                f"{rel.from_entity.title} → {rel.to_entity.title}"
                if rel.to_entity
                else f"{rel.from_entity.title}"
            )

            rel_content_stems = "\n".join(
                p for p in self._generate_variants(relation_title) if p and p.strip()
            )
            await self.repository.index_item(
                SearchIndexRow(
                    id=rel.id,
                    title=relation_title,
                    permalink=rel.permalink,
                    content_stems=rel_content_stems,
                    file_path=entity.file_path,
                    type=SearchItemType.RELATION.value,
                    entity_id=entity.id,
                    from_id=rel.from_id,
                    to_id=rel.to_id,
                    relation_type=rel.relation_type,
                    created_at=entity.created_at,
                    updated_at=entity.updated_at,
                    project_id=entity.project_id,
                )
            )

        # Also index in vector database if initialized
        if self._vector_initialized:
            try:
                # Index main entity content
                await self.index_entity_vector(entity)
                
                # Index observations in vector database
                for obs in entity.observations:
                    if obs.content:
                        obs_metadata = {
                            "entity_id": str(entity.id),
                            "project_id": str(entity.project_id),
                            "type": SearchItemType.OBSERVATION.value,
                            "title": f"{obs.category}: {obs.content[:100]}...",
                            "permalink": obs.permalink or "",
                            "file_path": entity.file_path or "",
                            "category": obs.category or "",
                            "tags": str(obs.tags) if obs.tags else "",
                            "created_at": entity.created_at.isoformat() if entity.created_at else "",
                            "updated_at": entity.updated_at.isoformat() if entity.updated_at else ""
                        }
                        
                        if self.vector_provider:
                            await self.vector_provider.index_document(
                                id=obs.permalink or f"obs_{obs.id}",
                                content=obs.content,
                                metadata=obs_metadata
                            )
                        logger.debug(f"Indexed observation {obs.id} in vector database")
                
                # Index relations in vector database
                for rel in entity.outgoing_relations:
                    relation_title = (
                        f"{rel.from_entity.title} → {rel.to_entity.title}"
                        if rel.to_entity
                        else f"{rel.from_entity.title}"
                    )
                    
                    rel_metadata = {
                        "entity_id": str(entity.id),
                        "project_id": str(entity.project_id),
                        "type": SearchItemType.RELATION.value,
                        "title": relation_title,
                        "permalink": rel.permalink or "",
                        "file_path": entity.file_path or "",
                        "from_id": str(rel.from_id) if rel.from_id else "",
                        "to_id": str(rel.to_id) if rel.to_id else "",
                        "relation_type": rel.relation_type or "",
                        "created_at": entity.created_at.isoformat() if entity.created_at else "",
                        "updated_at": entity.updated_at.isoformat() if entity.updated_at else ""
                    }
                    
                    if self.vector_provider:
                        await self.vector_provider.index_document(
                            id=rel.permalink or f"rel_{rel.id}",
                            content=relation_title,
                            metadata=rel_metadata
                        )
                    logger.debug(f"Indexed relation {rel.id} in vector database")
                    
            except Exception as e:
                logger.warning(f"Failed to index entity {entity.id} in vector database: {e}")

    async def delete_by_permalink(self, permalink: str):
        """Delete an item from the search index and vector database."""
        # Delete from FTS5 search index
        await self.repository.delete_by_permalink(permalink)
        
        # Also delete from vector database if initialized
        if self._vector_initialized and self.vector_provider:
            try:
                # Delete main document using permalink as document ID
                await self.vector_provider.delete_document(permalink)
                logger.debug(f"Deleted vector document with permalink: {permalink}")
                
                # Delete any potential chunks (they would have IDs like permalink#0, permalink#1, etc.)
                i = 0
                while True:
                    chunk_id = f"{permalink}#{i}"
                    deleted = await self.vector_provider.delete_document(chunk_id)
                    if not deleted:
                        break
                    logger.debug(f"Deleted vector chunk: {chunk_id}")
                    i += 1
                    
            except Exception as e:
                logger.warning(f"Failed to delete vector documents for permalink {permalink}: {e}")

    async def delete_by_entity_id(self, entity_id: int):
        """Delete an item from the search index and vector database."""
        # Delete from FTS5 search index
        await self.repository.delete_by_entity_id(entity_id)
        
        # Also delete from vector database if initialized
        if self._vector_initialized and self.vector_provider:
            try:
                await self.delete_entity_vector(entity_id)
            except Exception as e:
                logger.warning(f"Failed to delete vector index for entity {entity_id}: {e}")

    async def index_entity_vector(self, entity: Entity) -> None:
        """Index an entity in the vector database."""
        if not self._vector_initialized or not self.vector_provider or not self.text_chunker:
            return

        try:
            # Get entity content for vectorization
            content = await self.file_service.read_entity_content(entity)
            if not content:
                return

            # Prepare metadata for vector indexing
            metadata = {
                "entity_id": str(entity.id),
                "project_id": str(entity.project_id),
                "type": SearchItemType.ENTITY.value,
                "title": entity.title or "",
                "permalink": entity.permalink or "",
                "file_path": entity.file_path or "",
                "entity_type": entity.entity_type or "",
                "created_at": entity.created_at.isoformat() if entity.created_at else "",
                "updated_at": entity.updated_at.isoformat() if entity.updated_at else ""
            }

            # Chunk content if it's large
            if len(content) > self.text_chunker.chunk_size:
                chunks = self.text_chunker.chunk_text(content, metadata)
                
                # Index each chunk separately
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{entity.permalink or entity.id}#{i}"
                    chunk_metadata = chunk["metadata"]
                    
                    await self.vector_provider.index_document(
                        id=chunk_id,
                        content=chunk["content"],
                        metadata=chunk_metadata
                    )
                    
                    logger.debug(f"Indexed vector chunk {i+1}/{len(chunks)} for entity {entity.id}")
            else:
                # Index as single document
                document_id = entity.permalink or str(entity.id)
                await self.vector_provider.index_document(
                    id=document_id,
                    content=content,
                    metadata=metadata
                )
                
                logger.debug(f"Indexed vector document for entity {entity.id}")

        except Exception as e:
            logger.error(f"Failed to index entity {entity.id} in vector database: {e}")

    async def delete_entity_vector(self, entity_id: int) -> None:
        """Delete an entity from the vector database."""
        if not self._vector_initialized or not self.vector_provider:
            return

        try:
            # We need to find the entity to get its permalink and related data for deletion
            entity = await self.entity_repository.find_by_id(entity_id)
            
            # Determine the document ID (use permalink if entity exists, otherwise entity ID)
            document_id = (entity.permalink if entity and entity.permalink else str(entity_id))
            
            # Delete main entity document
            await self.vector_provider.delete_document(document_id)
            
            # Delete any entity chunks (they would have IDs like document_id#0, document_id#1, etc.)
            i = 0
            while True:
                chunk_id = f"{document_id}#{i}"
                deleted = await self.vector_provider.delete_document(chunk_id)
                if not deleted:
                    break
                i += 1
            
            # Delete observations from vector database
            if entity and entity.observations:
                for obs in entity.observations:
                    obs_id = obs.permalink or f"obs_{obs.id}"
                    await self.vector_provider.delete_document(obs_id)
                    logger.debug(f"Deleted observation {obs.id} from vector database")
            
            # Delete relations from vector database  
            if entity and entity.outgoing_relations:
                for rel in entity.outgoing_relations:
                    rel_id = rel.permalink or f"rel_{rel.id}"
                    await self.vector_provider.delete_document(rel_id)
                    logger.debug(f"Deleted relation {rel.id} from vector database")
            
            logger.debug(f"Deleted all vector documents for entity {entity_id}")
                
        except Exception as e:
            logger.warning(f"Failed to delete entity {entity_id} from vector database: {e}")

    async def handle_delete(self, entity: Entity):
        """Handle complete entity deletion from search index including observations and relations.

        This replicates the logic from sync_service.handle_delete() to properly clean up
        all search index entries for an entity and its related data.
        """
        logger.debug(
            f"Cleaning up search index for entity_id={entity.id}, file_path={entity.file_path}, "
            f"observations={len(entity.observations)}, relations={len(entity.outgoing_relations)}"
        )

        # Clean up search index - same logic as sync_service.handle_delete()
        permalinks = (
            [entity.permalink]
            + [o.permalink for o in entity.observations]
            + [r.permalink for r in entity.outgoing_relations]
        )

        logger.debug(
            f"Deleting search index entries for entity_id={entity.id}, "
            f"index_entries={len(permalinks)}"
        )

        for permalink in permalinks:
            if permalink:
                await self.delete_by_permalink(permalink)
            else:
                await self.delete_by_entity_id(entity.id)
