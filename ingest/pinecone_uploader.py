"""Pinecone uploader for RAG document chunks."""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Union

import pinecone

# Import from app directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.embeddings import embedding_client
from app.schema import DocumentChunk
from app.settings import settings

logger = logging.getLogger(__name__)


class PineconeUploader:
    """Uploader for RAG document chunks to Pinecone."""
    
    def __init__(
        self,
        api_key: str = os.environ.get("PINECONE_API_KEY", settings.pinecone_api_key),
        environment: str = os.environ.get("PINECONE_ENV", settings.pinecone_env),
        index_name: str = os.environ.get("PINECONE_INDEX", settings.pinecone_index),
        batch_size: int = 100,
    ):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.batch_size = batch_size
        self._index = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the Pinecone client."""
        if self._initialized:
            return
            
        # Initialize Pinecone
        pinecone.init(
            api_key=self.api_key,
            environment=self.environment,
        )
        
        # Get the index
        if self.index_name not in pinecone.list_indexes():
            logger.warning(
                f"Pinecone index {self.index_name} not found, "
                f"available indexes: {pinecone.list_indexes()}"
            )
        
        self._index = pinecone.Index(self.index_name)
        self._initialized = True
        
        logger.info(f"Pinecone uploader initialized: index={self.index_name}")
    
    def upload_chunks(
        self, 
        chunks: List[DocumentChunk], 
        namespace: Optional[str] = None
    ) -> Dict:
        """Upload document chunks to Pinecone."""
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Ensure all chunks have embeddings
        chunks_to_embed = []
        for chunk in chunks:
            if chunk.embedding is None:
                chunks_to_embed.append(chunk)
        
        # Generate embeddings for chunks that don't have them
        if chunks_to_embed:
            self._generate_embeddings(chunks_to_embed)
        
        # Prepare vectors for Pinecone
        vectors = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Skipping chunk {chunk.id} with no embedding")
                continue
                
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
        
        # Upload vectors in batches
        results = {"upserted_count": 0, "error_count": 0}
        
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i:i + self.batch_size]
            try:
                upsert_response = self._index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
                results["upserted_count"] += upsert_response.get("upserted_count", 0)
            except Exception as e:
                logger.error(f"Error upserting batch to Pinecone: {str(e)}")
                results["error_count"] += len(batch)
        
        elapsed = time.time() - start_time
        logger.info(
            f"Uploaded {results['upserted_count']} vectors to Pinecone in {elapsed:.2f}s "
            f"with {results['error_count']} errors"
        )
        
        return results
    
    def _generate_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings for chunks that don't have them."""
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        try:
            # Initialize embeddings client
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Initialize if needed
            if not embedding_client._initialized:
                loop.run_until_complete(embedding_client.initialize())
            
            # Generate embeddings
            embeddings = loop.run_until_complete(embedding_client.embed_batch(texts))
            loop.close()
            
            # Assign embeddings back to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Continue with whatever embeddings we have
