"""MIME-type dispatch to chunker for document processing."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import magic
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

# Import chunker from app directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.chunker import DocumentChunker
from app.embeddings import embedding_client
from app.schema import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Document loader with MIME-type dispatch to appropriate chunking logic."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def process_document(
        self,
        file_path: Union[str, Path],
        document_id: str,
        metadata: Optional[Dict] = None,
        embed_chunks: bool = True
    ) -> List[DocumentChunk]:
        """Process a document file into chunks with optional embedding."""
        start_time = time.time()
        
        # Ensure metadata includes basic information
        if not metadata:
            metadata = {}
        
        file_path_obj = Path(file_path)
        if "source" not in metadata:
            metadata["source"] = str(file_path_obj)
        
        if "title" not in metadata:
            metadata["title"] = file_path_obj.name
        
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now().isoformat()
        
        # Detect MIME type
        mime_type = self._detect_mime_type(file_path)
        metadata["mime_type"] = mime_type
        
        logger.info(f"Processing {document_id} ({mime_type}): {file_path}")
        
        # Process the file into chunks
        chunks = self.chunker.process_file(
            file_path=file_path,
            document_id=document_id,
            metadata=metadata
        )
        
        # Generate embeddings if requested
        if embed_chunks and chunks:
            self._embed_chunks(chunks)
        
        elapsed = time.time() - start_time
        logger.info(
            f"Document {document_id} processed in {elapsed:.2f}s: "
            f"{len(chunks)} chunks created"
        )
        
        return chunks
    
    def _detect_mime_type(self, file_path: Union[str, Path]) -> str:
        """Detect the MIME type of a file."""
        return magic.from_file(str(file_path), mime=True)
    
    def _embed_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings for all chunks."""
        import asyncio
        
        texts = [chunk.text for chunk in chunks]
        
        try:
            # Initialize embeddings client if needed
            if not embedding_client._initialized:
                asyncio.run(embedding_client.initialize())
            
            # Generate embeddings
            embeddings = asyncio.run(embedding_client.embed_batch(texts))
            
            # Assign embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Continue without embeddings, they'll be generated during upload

