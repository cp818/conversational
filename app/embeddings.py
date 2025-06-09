"""Optimized async Gemini embedding client with caching, batching, and retry logic."""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from app.settings import settings
from app.cache import cache_service

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Optimized client for the Gemini Embedding API with caching and retry logic."""
    
    def __init__(
        self,
        project_id: str = settings.gemini_project_id,
        location: str = settings.gemini_location,
        model: str = settings.gemini_embed_model,
    ):
        self.project_id = project_id
        self.location = location
        self.model = model
        self.batch_size = settings.embedding_batch_size
        self._http_client = None
        self._endpoint = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model}:predict"
        self._initialized = False
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._prefetch_cache = {}
        self._warm_connection = False
    
    async def initialize(self):
        """Initialize the embedding client asynchronously with optimized connection pool and prewarming."""
        if self._initialized:
            return
            
        # Create an optimized async HTTP client with enhanced connection pooling
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.embedding_timeout, connect=3.0),  # Reduced connect timeout
            limits=httpx.Limits(
                max_connections=settings.max_concurrent_requests * 2,  # Doubled max connections
                max_keepalive_connections=settings.max_concurrent_requests * 2,
                keepalive_expiry=60.0  # Increased keepalive
            ),
            http2=True,  # Enable HTTP/2 for multiplexing requests
        )
        
        # Initialize cache service
        await cache_service.initialize()
        
        # Pre-warm connection if enabled
        if not self._warm_connection:
            try:
                # Make a lightweight request to pre-establish connection
                await self._http_client.get(
                    f"https://{self.location}-aiplatform.googleapis.com/v1", 
                    timeout=2.0
                )
                self._warm_connection = True
            except Exception as e:
                logger.warning(f"Connection pre-warming failed (non-critical): {str(e)}")
        
        self._initialized = True
        logger.debug(f"Optimized Gemini embedding client initialized: {self.model}")
        
        # Start background task for cache maintenance
        if settings.use_cache:
            asyncio.create_task(self._maintain_cache())
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
    )
    async def _embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text with retry logic."""
        if not self._initialized:
            await self.initialize()
            
        # Prepare payload
        payload = {
            "instances": [{"content": text}]
        }
        
        start_time = time.time()
        
        # Make async request with semaphore to limit concurrent connections
        async with self._semaphore:
            try:
                response = await self._http_client.post(
                    self._endpoint,
                    json=payload,
                )
                
                if response.status_code != 200:
                    logger.error(f"Embedding error: {response.status_code} - {response.text}")
                    raise Exception(f"Failed to generate embedding: {response.text}")
                
                result = response.json()
                embedding = result.get("predictions", [])[0].get("embeddings", {}).get("values", [])
                
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > 100:  # Log slow requests
                    logger.warning(f"Slow embedding request: {elapsed_ms:.2f}ms")
                
                return embedding
                
            except httpx.TimeoutException:
                logger.error(f"Embedding request timed out after {settings.embedding_timeout}s")
                raise Exception("Embedding timed out")
                
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                raise
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text with caching."""
        if settings.use_cache:
            # Check cache first
            cache_key = f"embed:{cache_service.get_hash(text)}"
            cached = await cache_service.get(cache_key)
            if cached:
                logger.debug("Cache hit for embedding")
                return cached

        # Generate embedding
        start_time = time.time()
        embedding = await self._embed_single(text)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"Generated embedding in {elapsed_ms:.2f}ms")
        
        # Cache the result
        if settings.use_cache:
            await cache_service.set(cache_key, embedding)
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts efficiently with caching."""
        if not self._initialized:
            await self.initialize()
            
        result_embeddings = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache first for each text
        if settings.use_cache:
            for i, text in enumerate(texts):
                cache_key = f"embed:{cache_service.get_hash(text)}"
                cached = await cache_service.get(cache_key)
                if cached:
                    result_embeddings[i] = cached
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))
        
        if not texts_to_embed:
            return result_embeddings  # All from cache
            
        # Generate embeddings for texts not in cache
        batches = []
        batch_indices = []
        for i in range(0, len(texts_to_embed), self.batch_size):
            batch = texts_to_embed[i:i + self.batch_size]
            indices = indices_to_embed[i:i + self.batch_size]
            batches.append(batch)
            batch_indices.append(indices)
        
        start_time = time.time()
        for batch, indices in zip(batches, batch_indices):
            # Process each batch concurrently
            tasks = [self._embed_single(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions and retry failed items
            for i, embedding in enumerate(batch_embeddings):
                if isinstance(embedding, Exception):
                    logger.warning(f"Error embedding text: {embedding}, using zeros")
                    # Insert a zero vector of the expected dimension (768 for Gemini)
                    batch_embeddings[i] = [0.0] * 768
                
                # Store in results and cache
                orig_idx = indices[i]
                result_embeddings[orig_idx] = batch_embeddings[i]
                
                # Cache the result if not an error
                if settings.use_cache and not isinstance(embedding, Exception):
                    cache_key = f"embed:{cache_service.get_hash(texts_to_embed[indices[i]])}"
                    await cache_service.set(cache_key, batch_embeddings[i])
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"Generated {len(texts_to_embed)} embeddings in {elapsed_ms:.2f}ms")
        
        return result_embeddings
    
    async def _maintain_cache(self):
        """Background task for cache maintenance and prefetching strategies."""
        try:
            while True:
                # Periodically clean up the prefetch cache to prevent memory bloat
                if len(self._prefetch_cache) > 1000:
                    # Keep only the 100 most recently used items
                    self._prefetch_cache = dict(list(self._prefetch_cache.items())[-100:])
                
                # Sleep to avoid consuming resources
                await asyncio.sleep(60)  # Run maintenance every minute
        except asyncio.CancelledError:
            # Task was cancelled, clean exit
            pass
        except Exception as e:
            logger.warning(f"Cache maintenance task encountered an error: {str(e)}")
    
    async def prefetch_embeddings(self, texts: List[str]) -> None:
        """Prefetch embeddings for future use to reduce latency for large files."""
        if not settings.use_progressive_tokenization or len(texts) < 5:
            return

        try:
            # Identify key texts to prefetch (e.g., first paragraph of each section)
            # This is a simplified heuristic - in a real system this could be more sophisticated
            sample_texts = texts[::max(1, len(texts) // 10)][:5]  # Take ~5 evenly distributed samples
            
            # Create a background task to fetch embeddings that doesn't block
            asyncio.create_task(self._background_prefetch(sample_texts))
        except Exception as e:
            # Non-critical operation, just log errors
            logger.debug(f"Prefetch error (non-critical): {str(e)}")
    
    async def _background_prefetch(self, texts: List[str]) -> None:
        """Background task to prefetch embeddings without blocking."""
        try:
            if not self._initialized:
                await self.initialize()
                
            for text in texts:
                cache_key = f"embed:{cache_service.get_hash(text)}"
                
                # Skip if already in cache
                cached = await cache_service.get(cache_key)
                if cached:
                    continue
                    
                # Skip if already in prefetch cache
                if text in self._prefetch_cache:
                    continue
                
                # Mark as being prefetched
                self._prefetch_cache[text] = True
                
                # Generate embedding and cache it
                try:
                    embedding = await self._embed_single(text)
                    await cache_service.set(cache_key, embedding)
                except Exception:
                    # Errors in prefetching are non-critical
                    pass
        except Exception as e:
            logger.debug(f"Background prefetch error (non-critical): {str(e)}")
    
    def normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit vectors for more accurate similarity search."""
        normalized = []
        for emb in embeddings:
            # Convert to numpy for efficient computation
            emb_array = np.array(emb)
            # Compute L2 norm
            norm = np.linalg.norm(emb_array)
            # Normalize vector
            if norm > 0:
                normalized_emb = emb_array / norm
            else:
                normalized_emb = emb_array
            # Convert back to list
            normalized.append(normalized_emb.tolist())
        return normalized


# Create a singleton instance
embedding_client = EmbeddingClient()
