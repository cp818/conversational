"""Two-stage semantic + keyword retriever for RAG."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union

import httpx
import numpy as np
import pinecone
from elasticsearch import AsyncElasticsearch

from app.embeddings import embedding_client
from app.schema import DocumentChunk
from app.settings import settings

logger = logging.getLogger(__name__)


class RetrieverService:
    """Two-stage retriever combining vector search with optional keyword search."""
    
    def __init__(self):
        self._pinecone_index = None
        self._elastic_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the retriever clients asynchronously."""
        if self._initialized:
            return
        
        # Initialize embedding client
        await embedding_client.initialize()
        
        # Initialize Pinecone
        pinecone.init(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_env,
        )
        
        if settings.pinecone_index not in pinecone.list_indexes():
            logger.warning(f"Pinecone index {settings.pinecone_index} not found")
        
        self._pinecone_index = pinecone.Index(settings.pinecone_index)
        
        # Initialize Elasticsearch (if configured)
        if settings.elastic_cloud_id and settings.elastic_api_key:
            self._elastic_client = AsyncElasticsearch(
                cloud_id=settings.elastic_cloud_id,
                api_key=settings.elastic_api_key,
            )
        
        self._initialized = True
        logger.info("Retriever service initialized")
    
    async def retrieve(
        self,
        query: str,
        top_k: int = settings.top_k_results,
        filters: Optional[Dict] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict]:
        """Perform optimized parallel retrieval with semantic and keyword search."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Prepare retrieval tasks concurrently
        tasks = []
        
        # Task 1: Generate query embedding - always needed
        embed_task = asyncio.create_task(embedding_client.embed(query))
        tasks.append(embed_task)
        
        # Prioritize initial vector search results if TTFT is critical
        if settings.prioritize_ttft:
            # First, get a smaller number of quick results
            initial_k = max(1, top_k // 2)  # Get at least one result, but fewer than requested
            pinecone_filter = self._prepare_pinecone_filters(filters, namespace)
            
            # Create task for initial fast vector search with reduced top_k
            initial_results_task = asyncio.create_task(
                self._fast_vector_search(embed_task, initial_k, pinecone_filter)
            )
            tasks.append(initial_results_task)
        
        # Now prepare the main retrieval tasks
        if settings.use_parallel_retrieval and filters and self._elastic_client:
            # Prepare Pinecone filter once
            pinecone_filter = self._prepare_pinecone_filters(filters, namespace)
            
            # Set up parallel tasks for vector and keyword search
            vector_task = asyncio.create_task(
                self._vector_search(await embed_task, top_k, pinecone_filter)
            )
            elastic_task = asyncio.create_task(
                self._keyword_search(query, filters, top_k, namespace)
            )
            
            # Wait for both to complete
            vector_results, elastic_results = await asyncio.gather(vector_task, elastic_task)
            
            # Combine results
            results = await self._combine_results(vector_results, elastic_results, top_k)
        else:
            # Just do vector search if no keyword search or parallel retrieval disabled
            pinecone_filter = self._prepare_pinecone_filters(filters, namespace)
            results = await self._vector_search(await embed_task, top_k, pinecone_filter)
        
        total_time = time.time() - start_time
        latency_ms = int(total_time * 1000)
        logger.debug(f"Total retrieval latency: {latency_ms}ms for query: {query[:30]}...")
        
        # Track latency metrics for monitoring
        if latency_ms > 850:
            logger.warning(f"High retrieval latency: {latency_ms}ms exceeds target of 850ms")
        
        return results
        
    async def _fast_vector_search(
        self, embed_task: asyncio.Task, top_k: int, filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Optimized vector search for low latency first results."""
        try:
            # Wait for embedding task to complete
            query_embedding = await embed_task
            
            # Simplified query params focused on speed
            query_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
            }
            
            if filters:
                query_params["filter"] = filters
            
            # Execute Pinecone query with shorter timeout
            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._pinecone_index.query(**query_params)),
                timeout=settings.pinecone_timeout
            )
            
            # Process results
            matches = results.get("matches", [])
            
            # Convert to a consistent format - minimal processing for speed
            processed_results = []
            for match in matches:
                processed_results.append({
                    "id": match["id"],
                    "text": match.get("metadata", {}).get("text", ""),
                    "score": float(match["score"]),
                    "source": match.get("metadata", {}).get("source", ""),
                    "metadata": match.get("metadata", {}),
                })
            
            return processed_results
            
        except asyncio.TimeoutError:
            logger.warning("Fast vector search timed out, falling back to default results")
            return []  # Return empty results, full search will complete later
        except Exception as e:
            logger.error(f"Fast vector search error: {str(e)}")
            return []
    
    def _prepare_pinecone_filters(
        self, filters: Optional[Dict], namespace: Optional[str]
    ) -> Dict:
        """Prepare filters for Pinecone query."""
        pinecone_filter = {}
        
        if filters:
            # Convert api-friendly filters to Pinecone format
            meta_filters = {}
            for key, value in filters.items():
                if key.startswith("metadata."):
                    # Extract the actual metadata field name
                    field_name = key[len("metadata."):]
                    meta_filters[field_name] = value
                else:
                    # Direct filter
                    meta_filters[key] = value
            
            if meta_filters:
                pinecone_filter = {"metadata": meta_filters}
        
        return pinecone_filter
    
    async def _vector_search(
        self, query_embedding: List[float], top_k: int, filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Perform vector search using Pinecone."""
        try:
            # We use the synchronous Pinecone API wrapped in an asyncio executor
            # to avoid blocking the event loop
            query_params = {
                "vector": query_embedding,
                "top_k": top_k * 2,  # Get more results for reranking
                "include_metadata": True,
            }
            
            if filters:
                query_params["filter"] = filters
            
            # Execute Pinecone query in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: self._pinecone_index.query(**query_params)
            )
            
            # Process results
            matches = results.get("matches", [])
            
            # Convert to a consistent format
            processed_results = []
            for match in matches:
                processed_results.append({
                    "id": match["id"],
                    "text": match.get("metadata", {}).get("text", ""),
                    "score": float(match["score"]),
                    "source": match.get("metadata", {}).get("source", ""),
                    "metadata": match.get("metadata", {}),
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            return []
    
    async def _keyword_search(
        self, query: str, filters: Dict, top_k: int, index: Optional[str] = None
    ) -> List[Dict]:
        """Perform keyword search using Elasticsearch."""
        if not self._elastic_client:
            return []
        
        try:
            # Build Elasticsearch query
            es_index = index if index else "documents"
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"content": query}}
                        ],
                        "filter": []
                    }
                },
                "size": top_k
            }
            
            # Add filters
            for key, value in filters.items():
                if isinstance(value, list):
                    es_query["query"]["bool"]["filter"].append(
                        {"terms": {key: value}}
                    )
                else:
                    es_query["query"]["bool"]["filter"].append(
                        {"term": {key: value}}
                    )
            
            # Execute search
            response = await self._elastic_client.search(index=es_index, body=es_query)
            
            # Process results
            hits = response.get("hits", {}).get("hits", [])
            results = []
            
            for hit in hits:
                results.append({
                    "id": hit["_id"],
                    "text": hit["_source"].get("content", ""),
                    "score": float(hit["_score"]),
                    "source": hit["_source"].get("source", ""),
                    "metadata": {
                        k: v for k, v in hit["_source"].items() 
                        if k not in ["content", "embedding"]
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search error: {str(e)}")
            return []
    
    async def _combine_results(
        self, vector_results: List[Dict], keyword_results: List[Dict], top_k: int
    ) -> List[Dict]:
        """Combine and rerank results from vector and keyword search."""
        # Simple combination: give vector results higher weight
        combined = {}
        
        # Process vector results (higher weight)
        for i, result in enumerate(vector_results):
            result_id = result["id"]
            if result_id not in combined:
                combined[result_id] = result.copy()
                # Normalize score and apply weight (0.7 for vector)
                combined[result_id]["score"] = result["score"] * 0.7
            
        # Process keyword results (lower weight)
        for i, result in enumerate(keyword_results):
            result_id = result["id"]
            if result_id in combined:
                # Add normalized keyword score
                combined[result_id]["score"] += result["score"] * 0.3
            else:
                # New result from keyword search
                combined[result_id] = result.copy()
                combined[result_id]["score"] = result["score"] * 0.3
        
        # Sort by score and limit to top_k
        sorted_results = sorted(
            combined.values(), key=lambda x: x["score"], reverse=True
        )
        
        return sorted_results[:top_k]
    
    async def upsert_chunks(
        self, chunks: List[DocumentChunk], namespace: Optional[str] = None
    ) -> Dict:
        """Upsert document chunks to the vector database."""
        if not self._initialized:
            await self.initialize()
        
        # Check if chunks already have embeddings
        chunks_to_embed = []
        texts_to_embed = []
        
        for chunk in chunks:
            if not chunk.embedding:
                chunks_to_embed.append(chunk)
                texts_to_embed.append(chunk.text)
        
        # Generate embeddings if needed
        if texts_to_embed:
            embeddings = await embedding_client.embed_batch(texts_to_embed)
            
            # Add embeddings back to chunks
            for i, chunk in enumerate(chunks_to_embed):
                chunk.embedding = embeddings[i]
        
        # Prepare vectors for Pinecone
        vectors = []
        for chunk in chunks:
            vectors.append({
                "id": chunk.id,
                "values": chunk.embedding,
                "metadata": {
                    "text": chunk.text[:1000],  # Limit text size in metadata
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.metadata.source,
                    "mime_type": chunk.metadata.mime_type,
                    "created_at": chunk.metadata.created_at,
                    "title": chunk.metadata.title or "",
                }
            })
        
        # Upsert in batches of 100
        batch_size = 100
        results = {"upserted_count": 0, "error_count": 0}
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                # Execute Pinecone upsert in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                upsert_response = await loop.run_in_executor(
                    None, 
                    lambda: self._pinecone_index.upsert(
                        vectors=batch,
                        namespace=namespace
                    )
                )
                results["upserted_count"] += upsert_response.get("upserted_count", 0)
            except Exception as e:
                logger.error(f"Error upserting batch to Pinecone: {str(e)}")
                results["error_count"] += len(batch)
        
        # TODO: Add parallel indexing to Elasticsearch if configured
        
        logger.info(
            f"Upserted {results['upserted_count']} chunks with "
            f"{results['error_count']} errors"
        )
        
        return results
    
    async def close(self):
        """Close all clients."""
        if self._elastic_client:
            await self._elastic_client.close()
        
        await embedding_client.close()


# Singleton instance
retriever_service = RetrieverService()
