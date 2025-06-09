"""Application settings using pydantic-settings with environment variables."""

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Gemini API settings
    gemini_project_id: str = Field(..., description="GCP Project ID for Vertex AI Gemini API")
    gemini_location: str = Field("us-central1", description="GCP region for Vertex AI")
    gemini_embed_model: str = Field("gemini-embedding-001", description="Gemini embedding model")
    gemini_llm_model: str = Field("models/gemini-2.5-flash", description="Gemini LLM model")
    
    # Pinecone settings
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_env: str = Field("us-central1-gcp", description="Pinecone environment")
    pinecone_index: str = Field("rag-index", description="Pinecone index name")
    pinecone_namespace: str = Field("default", description="Pinecone namespace")
    pinecone_timeout: float = Field(0.3, description="Timeout for Pinecone queries in seconds")
    
    # Elastic settings (optional)
    elastic_cloud_id: str = Field(None, description="Elastic Cloud ID")
    elastic_api_key: str = Field(None, description="Elastic API key")
    elastic_timeout: float = Field(0.3, description="Timeout for Elastic queries in seconds")
    
    # RAG settings
    chunk_size_tokens: int = Field(512, description="Target chunk size in tokens")
    chunk_overlap_tokens: int = Field(64, description="Chunk overlap in tokens")
    top_k_results: int = Field(3, description="Number of results to retrieve - reduced for latency")
    context_max_bytes: int = Field(4096, description="Maximum context size in bytes - reduced for latency")
    similarity_threshold: float = Field(0.75, description="Minimum similarity score to include results")
    
    # Performance settings
    embedding_batch_size: int = Field(16, description="Batch size for embedding requests")
    embedding_timeout: float = Field(0.5, description="Timeout for embedding requests in seconds")
    max_concurrent_requests: int = Field(20, description="Maximum concurrent requests")
    max_workers: int = Field(4, description="Number of worker threads for parallel operations")
    
    # Caching settings
    use_cache: bool = Field(True, description="Enable caching")
    use_redis: bool = Field(False, description="Use Redis for distributed caching")
    redis_uri: str = Field("", description="Redis connection URI")
    cache_ttl_seconds: int = Field(300, description="Cache TTL in seconds")
    max_cache_items: int = Field(10000, description="Maximum items in local memory cache")
    
    # App settings
    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("WARNING", description="Log level - raised to reduce overhead")
    port: int = Field(8080, description="Application port")
    min_instances: int = Field(3, description="Minimum number of Cloud Run instances - increased for availability")
    
    # Advanced optimization settings
    use_parallel_retrieval: bool = Field(True, description="Enable parallel retrieval from vector and keyword stores")
    prioritize_ttft: bool = Field(True, description="Prioritize time to first token over comprehensive results")
    use_progressive_tokenization: bool = Field(True, description="Use progressive tokenization for large files")
    
    @validator("elastic_cloud_id", "elastic_api_key", pre=True)
    def validate_optional(cls, v):
        """Allow empty strings to be converted to None for optional fields."""
        if v == "":
            return None
        return v
    
    @validator("redis_uri", pre=True)
    def validate_redis_uri(cls, v, values):
        """Only require Redis URI if use_redis is True."""
        if values.get("use_redis", False) and not v:
            raise ValueError("Redis URI is required when use_redis is True")
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Create a global settings instance
settings = Settings()
