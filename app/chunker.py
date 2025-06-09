"""Universal document loader and chunker using unstructured and Apache Tika."""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import magic
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Text
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

from app.schema import DocumentChunk, DocumentMetadata
from app.settings import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Universal document loader and chunker for various file formats."""
    
    def __init__(
        self,
        chunk_size: int = settings.chunk_size_tokens,
        chunk_overlap: int = settings.chunk_overlap_tokens,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _get_mime_type(self, file_path: Union[str, Path]) -> str:
        """Detect the MIME type of a file."""
        return magic.from_file(file_path, mime=True)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text (rough approximation)."""
        # A simple approximation: ~4 chars = 1 token for English
        return len(text) // 4
    
    def _extract_elements(
        self, file_path: Union[str, Path], mime_type: Optional[str] = None
    ) -> List[Element]:
        """Extract elements from a document based on its MIME type."""
        if not mime_type:
            mime_type = self._get_mime_type(file_path)
        
        logger.info(f"Processing file: {file_path} with MIME type: {mime_type}")
        
        if mime_type.startswith("application/pdf"):
            # Use PDF-specific partitioning with OCR if needed
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                extract_images_in_pdf=False,
                use_ocr_for_pages=True,
            )
        else:
            # Use automatic partitioning for other document types
            elements = partition(
                filename=file_path,
                include_page_breaks=True,
                include_metadata=True,
            )
        
        return elements
    
    def _split_text_into_chunks(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        # Convert token sizes to approximate character counts
        char_size = chunk_size * 4
        char_overlap = chunk_overlap * 4
        
        if len(text) <= char_size:
            return [text]
        
        chunks = []
        for i in range(0, len(text), char_size - char_overlap):
            chunk = text[i:i + char_size]
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunks_from_elements(
        self, elements: List[Element], document_id: str, metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """Convert elements to document chunks."""
        chunks = []
        chunk_index = 0
        
        # Join text elements that are small enough to be combined
        buffer = ""
        current_token_count = 0
        
        for element in elements:
            if not isinstance(element, Text):
                # Process non-text elements if needed
                continue
                
            element_text = element.text.strip()
            if not element_text:
                continue
                
            element_token_count = self._estimate_tokens(element_text)
            
            # If adding this element would exceed chunk size, create a chunk and reset buffer
            if current_token_count + element_token_count > self.chunk_size and buffer:
                chunk = DocumentChunk(
                    id=f"{document_id}-{chunk_index}",
                    text=buffer,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    metadata=metadata,
                    embedding=None,
                )
                chunks.append(chunk)
                chunk_index += 1
                buffer = element_text
                current_token_count = element_token_count
            else:
                # Add to buffer with a space if needed
                if buffer:
                    buffer += " "
                buffer += element_text
                current_token_count += element_token_count
        
        # Add the remaining text as a chunk
        if buffer:
            chunk = DocumentChunk(
                id=f"{document_id}-{chunk_index}",
                text=buffer,
                document_id=document_id,
                chunk_index=chunk_index,
                metadata=metadata,
                embedding=None,
            )
            chunks.append(chunk)
        
        return chunks
    
    async def process_file(
        self, file_path: Union[str, Path], document_id: str, metadata: Optional[Dict] = None,
        max_file_size_mb: int = 100
    ) -> List[DocumentChunk]:
        """Process a file into chunks with optimizations for large files."""
        start_time = time.time()
        mime_type = self._get_mime_type(file_path)
        
        # Check file size and apply optimizations for large files
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        logger.debug(f"Processing file {file_path} ({file_size_mb:.2f} MB)")
        
        if not metadata:
            metadata = {}
        
        doc_metadata = DocumentMetadata(
            source=str(file_path),
            mime_type=mime_type,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **metadata
        )
        
        # For extremely large files, use a more efficient processing approach
        if file_size_mb > max_file_size_mb and settings.use_progressive_tokenization:
            logger.warning(f"Large file detected ({file_size_mb:.2f}MB), using progressive chunking")
            chunks = await self._process_large_file(file_path, mime_type, document_id, doc_metadata)
        else:
            # Standard processing for normal-sized files
            loop = asyncio.get_event_loop()
            # Run CPU-intensive extraction in a thread pool
            elements = await loop.run_in_executor(None, self._extract_elements, file_path, mime_type)
            chunks = self._create_chunks_from_elements(elements, document_id, doc_metadata)
        
        elapsed = time.time() - start_time
        logger.info(
            f"Processed document {document_id}: {len(chunks)} chunks created in {elapsed:.2f}s"
        )
        
        # Prefetch embeddings for a sample of chunks to warm up caches
        if len(chunks) > 5 and hasattr(settings, 'use_progressive_tokenization') and settings.use_progressive_tokenization:
            from app.embeddings import embedding_client
            sample_texts = [chunk.text for chunk in chunks[::max(1, len(chunks)//5)][:5]]
            await embedding_client.prefetch_embeddings(sample_texts)
        
        return chunks
    
    async def _process_large_file(
        self, file_path: Union[str, Path], mime_type: str, document_id: str, metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """Process large files in a memory-efficient way with streaming."""
        chunks = []
        chunk_index = 0
        
        try:
            # For large PDFs, process page by page
            if mime_type.startswith("application/pdf"):
                # Use tika for more memory-efficient processing
                import subprocess
                from tika import parser
                
                # Process the PDF in batches of pages to avoid memory issues
                with tempfile.TemporaryDirectory() as temp_dir:
                    # First extract total page count
                    page_info = subprocess.run(
                        ["pdfinfo", str(file_path)], 
                        capture_output=True, 
                        text=True
                    )
                    
                    # Parse output to get page count
                    page_count = 0
                    for line in page_info.stdout.split("\n"):
                        if line.startswith("Pages:"):
                            page_count = int(line.split(":")[1].strip())
                            break
                    
                    logger.info(f"Large PDF with {page_count} pages detected")
                    
                    # Process in batches of 10 pages
                    batch_size = 10
                    for batch_start in range(0, page_count, batch_size):
                        batch_end = min(batch_start + batch_size, page_count)
                        logger.debug(f"Processing PDF pages {batch_start+1}-{batch_end}")
                        
                        # Extract this batch of pages
                        batch_text = ""
                        for page in range(batch_start, batch_end):
                            # Parse page text with tika
                            parsed = parser.from_file(
                                str(file_path), 
                                requestOptions={'pages': f"{page+1}-{page+1}"}
                            )
                            page_text = parsed.get('content', "").strip()
                            if page_text:
                                batch_text += f"Page {page+1}:\n{page_text}\n\n"
                        
                        # Create chunks from the batch text
                        text_chunks = self._split_text_into_chunks(
                            batch_text, self.chunk_size, self.chunk_overlap
                        )
                        
                        # Create DocumentChunk objects
                        for text_chunk in text_chunks:
                            chunk = DocumentChunk(
                                id=f"{document_id}-{chunk_index}",
                                text=text_chunk,
                                document_id=document_id,
                                chunk_index=chunk_index,
                                metadata=metadata,
                                embedding=None,
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                            
                        # Force garbage collection to free memory
                        import gc
                        gc.collect()
            else:
                # For other large files, use a streaming approach with unstructured
                # Process the file in smaller chunks to avoid memory issues
                loop = asyncio.get_event_loop()
                
                # We'll read and process the file in sections
                with open(file_path, 'rb') as file:
                    file_content = file.read()
                
                # Process in batches of ~5MB
                batch_size = 5 * 1024 * 1024  # 5MB
                for i in range(0, len(file_content), batch_size):
                    batch = file_content[i:i+batch_size]
                    
                    # Create a temporary file for this batch
                    with tempfile.NamedTemporaryFile(suffix=Path(file_path).suffix, delete=False) as temp_file:
                        temp_file.write(batch)
                        temp_path = temp_file.name
                    
                    try:
                        # Process this batch in a thread to avoid blocking
                        elements = await loop.run_in_executor(
                            None, self._extract_elements, temp_path, mime_type
                        )
                        
                        # Create chunks from elements
                        batch_chunks = self._create_chunks_from_elements(
                            elements, document_id, metadata, start_index=chunk_index
                        )
                        
                        chunks.extend(batch_chunks)
                        chunk_index += len(batch_chunks)
                        
                        # Update progress
                        logger.debug(
                            f"Processed {i+len(batch)}/{len(file_content)} bytes, "
                            f"created {len(batch_chunks)} chunks"
                        )
                    finally:
                        # Clean up temp file
                        try:
                            Path(temp_path).unlink()
                        except Exception:
                            pass
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing large file: {str(e)}")
            # Fall back to standard processing if progressive approach fails
            loop = asyncio.get_event_loop()
            elements = await loop.run_in_executor(None, self._extract_elements, file_path, mime_type)
            return self._create_chunks_from_elements(elements, document_id, metadata)
    
    def _create_chunks_from_elements(
        self, elements: List[Element], document_id: str, metadata: DocumentMetadata, start_index: int = 0
    ) -> List[DocumentChunk]:
        """Convert elements to document chunks."""
        chunks = []
        chunk_index = start_index
        
        # Join text elements that are small enough to be combined
        buffer = ""
        current_token_count = 0
        
        for element in elements:
            if not isinstance(element, Text):
                # Process non-text elements if needed
                continue
                
            element_text = element.text.strip()
            if not element_text:
                continue
                
            element_token_count = self._estimate_tokens(element_text)
            
            # If adding this element would exceed chunk size, create a chunk and reset buffer
            if current_token_count + element_token_count > self.chunk_size and buffer:
                chunk = DocumentChunk(
                    id=f"{document_id}-{chunk_index}",
                    text=buffer,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    metadata=metadata,
                    embedding=None,
                )
                chunks.append(chunk)
                chunk_index += 1
                buffer = element_text
                current_token_count = element_token_count
            else:
                # Add to buffer with a space if needed
                if buffer:
                    buffer += " "
                buffer += element_text
                current_token_count += element_token_count
        
        # Add the remaining text as a chunk
        if buffer:
            chunk = DocumentChunk(
                id=f"{document_id}-{chunk_index}",
                text=buffer,
                document_id=document_id,
                chunk_index=chunk_index,
                metadata=metadata,
                embedding=None,
            )
            chunks.append(chunk)
        
        return chunks
    
    async def process_text(
        self, text: str, document_id: str, metadata: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        """Process raw text into chunks."""
        if not metadata:
            metadata = {}
        
        doc_metadata = DocumentMetadata(
            source="text_input",
            mime_type="text/plain",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **metadata
        )
        
        # Split text into appropriate chunks
        text_chunks = self._split_text_into_chunks(
            text, self.chunk_size, self.chunk_overlap
        )
        
        # Create DocumentChunk objects
        chunks = []
        for i, text_chunk in enumerate(text_chunks):
            chunk = DocumentChunk(
                id=f"{document_id}-{i}",
                text=text_chunk,
                document_id=document_id,
                chunk_index=i,
                metadata=doc_metadata,
                embedding=None,
            )
            chunks.append(chunk)
        
        # Prefetch embeddings to improve latency on subsequent requests
        if len(chunks) > 5 and settings.use_progressive_tokenization:
            from app.embeddings import embedding_client
            sample_texts = [chunks[0].text]  # Always prefetch first chunk
            if len(chunks) > 1:
                sample_texts.append(chunks[-1].text)  # And last chunk
            await embedding_client.prefetch_embeddings(sample_texts)
        
        logger.info(f"Processed text {document_id}: {len(chunks)} chunks created")
        return chunks


# Singleton instance
document_chunker = DocumentChunker()
