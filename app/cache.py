"""In-memory and Redis caching for RAG components."""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import aioredis

from app.settings import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Cache service for RAG components supporting both in-memory and Redis."""

    def __init__(self):
        self._redis = None
        self._initialized = False
        self._in_memory_cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize Redis connection pool if Redis is available."""
        if self._initialized:
            return

        if settings.use_redis and settings.redis_uri:
            try:
                self._redis = await aioredis.from_url(
                    settings.redis_uri,
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info(f"Connected to Redis at {settings.redis_uri}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {str(e)}")
                self._redis = None

        self._initialized = True

    async def get(self, key: str, ttl_seconds: int = 600) -> Optional[Any]:
        """Get value from cache (Redis or in-memory)."""
        if not self._initialized:
            await self.initialize()

        # Try Redis first if available
        if self._redis:
            try:
                cached_value = await self._redis.get(key)
                if cached_value:
                    return json.loads(cached_value)
            except Exception as e:
                logger.warning(f"Redis get error: {str(e)}")

        # Fall back to in-memory cache
        async with self._lock:
            if key in self._in_memory_cache:
                value, expiry = self._in_memory_cache[key]
                if expiry > time.time():
                    return value
                else:
                    # Expired
                    del self._in_memory_cache[key]
        
        return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 600) -> bool:
        """Set value in cache with TTL."""
        if not self._initialized:
            await self.initialize()

        serialized = json.dumps(value)
            
        # Try Redis first if available
        if self._redis:
            try:
                await self._redis.set(key, serialized, ex=ttl_seconds)
                return True
            except Exception as e:
                logger.warning(f"Redis set error: {str(e)}")
        
        # Fall back to in-memory cache
        async with self._lock:
            self._in_memory_cache[key] = (value, time.time() + ttl_seconds)
            
            # Simple cache eviction if too large (keep it under control)
            if len(self._in_memory_cache) > settings.max_cache_items:
                # Remove expired items first
                current_time = time.time()
                expired_keys = [k for k, (_, exp) in self._in_memory_cache.items() if exp <= current_time]
                for k in expired_keys:
                    del self._in_memory_cache[k]
                
                # If still too large, remove oldest items
                if len(self._in_memory_cache) > settings.max_cache_items:
                    sorted_keys = sorted(self._in_memory_cache.items(), key=lambda x: x[1][1])
                    # Remove oldest 10% of items
                    to_remove = int(len(sorted_keys) * 0.1)
                    for k, _ in sorted_keys[:to_remove]:
                        del self._in_memory_cache[k]
        
        return True

    def get_hash(self, data: Union[str, Dict, List]) -> str:
        """Generate a consistent hash for data to use as cache key."""
        if isinstance(data, (dict, list)):
            serialized = json.dumps(data, sort_keys=True)
        else:
            serialized = str(data)
            
        return hashlib.md5(serialized.encode()).hexdigest()


# Global cache service instance
cache_service = CacheService()
