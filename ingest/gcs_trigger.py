"""Cloud Functions v2 entrypoint for processing documents uploaded to GCS."""

import base64
import json
import logging
import os
import time
from typing import Dict, Any

import functions_framework
from google.cloud import storage

from ingest.loader import DocumentLoader
from ingest.pinecone_uploader import PineconeUploader

# Configure logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def process_document(cloud_event):
    """Cloud Functions entry point for processing new documents in GCS."""
    start_time = time.time()
    
    try:
        # Parse the Cloud Event data
        data = cloud_event.data
        
        # Extract bucket and file info from the event
        bucket_name = data["bucket"]
        file_name = data["name"]
        
        logger.info(f"Processing new document: gs://{bucket_name}/{file_name}")
        
        # Skip if this is a deletion event (eventType = OBJECT_DELETE)
        if cloud_event.type == "google.cloud.storage.object.v1.deleted":
            logger.info(f"Skipping delete event for {file_name}")
            return {"status": "skipped", "reason": "delete event"}
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        # Generate a document ID from the file path
        document_id = _generate_document_id(file_name)
        
        # Download the file to a temporary location
        _, temp_local_path = _download_blob(blob)
        
        # Process the document
        result = process_local_document(
            document_id=document_id,
            file_path=temp_local_path,
            source_uri=f"gs://{bucket_name}/{file_name}",
        )
        
        # Clean up the temporary file
        os.remove(temp_local_path)
        
        # Add processing stats
        elapsed_time = time.time() - start_time
        result["processing_time_seconds"] = elapsed_time
        
        logger.info(
            f"Document {document_id} processed successfully in {elapsed_time:.2f}s: "
            f"{result.get('chunks_count', 0)} chunks"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}


def process_local_document(document_id: str, file_path: str, source_uri: str) -> Dict[str, Any]:
    """Process a document from a local file path."""
    try:
        # Initialize document loader
        loader = DocumentLoader()
        
        # Process the document
        chunks = loader.process_document(
            file_path=file_path,
            document_id=document_id,
            metadata={"source": source_uri}
        )
        
        # Initialize Pinecone uploader
        uploader = PineconeUploader()
        
        # Upload chunks to Pinecone
        upload_result = uploader.upload_chunks(chunks, namespace=document_id)
        
        # TODO: In parallel, store document metadata in Elasticsearch if configured
        
        return {
            "status": "success",
            "document_id": document_id,
            "chunks_count": len(chunks),
            "source_uri": source_uri,
            "upload_result": upload_result
        }
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}", exc_info=True)
        raise


def _download_blob(blob):
    """Download blob to a temporary file and return the file path."""
    import tempfile
    
    # Create a temporary file
    _, temp_path = tempfile.mkstemp()
    
    # Download the blob to the temporary file
    blob.download_to_filename(temp_path)
    
    logger.info(f"Downloaded {blob.name} to {temp_path}")
    
    return blob.content_type, temp_path


def _generate_document_id(file_path: str) -> str:
    """Generate a deterministic document ID from file path."""
    # Use base64 encoding of the path as ID, but make it URL-safe
    doc_id = base64.urlsafe_b64encode(file_path.encode()).decode().strip("=")
    return doc_id[:40]  # Truncate to a reasonable length


# For local testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python gcs_trigger.py <bucket_name> <file_name>")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    file_name = sys.argv[2]
    
    # Create a mock cloud event
    mock_event = type("CloudEvent", (), {
        "data": {"bucket": bucket_name, "name": file_name},
        "type": "google.cloud.storage.object.v1.finalized"
    })
    
    # Process the document
    result = process_document(mock_event)
    
    # Print the result
    print(json.dumps(result, indent=2))
