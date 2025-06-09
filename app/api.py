"""FastAPI routes and SSE/WebSocket handlers for RAG service."""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse

from app.embeddings import embedding_client
from app.generators import gemini_stream
from app.retrieval import retriever_service
from app.schema import (ChatRequest, HealthStatus, IngestRequest, IngestResponse,
                        QueryRequest, QueryResponse, StreamingMessage)
from app.settings import settings

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Microservice",
    description="Streaming Retrieval-Augmented Generation microservice for conversational AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=128)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add OpenTelemetry instrumentation
FastAPIInstrumentor.instrument_app(app)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Pre-initialize services to reduce cold-start latency
    await asyncio.gather(
        embedding_client.initialize(),
        retriever_service.initialize(),
        gemini_stream.initialize()
    )
    logger.info("All services initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    await asyncio.gather(
        embedding_client.close(),
        retriever_service.close(),
        gemini_stream.close()
    )
    logger.info("All services closed")


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    components = {
        "embeddings": True,
        "retriever": True,
        "generator": True,
    }
    
    # Check if services are initialized
    if not embedding_client._initialized:
        components["embeddings"] = False
    
    if not retriever_service._initialized:
        components["retriever"] = False
    
    if not gemini_stream._initialized:
        components["generator"] = False
    
    # Return health status
    return HealthStatus(
        status="healthy" if all(components.values()) else "degraded",
        version="1.0.0",
        components=components,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Retrieve relevant chunks based on a query."""
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        # Retrieve chunks
        results = await retriever_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            namespace=request.namespace,
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log the query
        logger.info(
            f"Query processed: id={query_id}, query='{request.query[:50]}...', "
            f"results={len(results)}, latency={latency_ms:.2f}ms"
        )
        
        # Return results
        return QueryResponse(
            chunks=results,
            query_id=query_id,
            total_latency_ms=latency_ms,
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest a document for RAG retrieval."""
    try:
        from app.chunker import document_chunker
        
        # Process the document into chunks
        chunks = document_chunker.process_text(
            text=request.content,
            document_id=request.document_id,
            metadata=request.metadata.dict(),
        )
        
        # Index the chunks
        result = await retriever_service.upsert_chunks(
            chunks=chunks,
            namespace=request.document_id,
        )
        
        return IngestResponse(
            document_id=request.document_id,
            chunks_count=len(chunks),
            success=True,
            message=f"Successfully ingested {len(chunks)} chunks",
        )
        
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        return IngestResponse(
            document_id=request.document_id,
            chunks_count=0,
            success=False,
            message=f"Error: {str(e)}",
        )


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming chat completion with RAG."""
    await websocket.accept()
    
    try:
        # Receive and parse the request
        request_data = await websocket.receive_text()
        request_dict = json.loads(request_data)
        request = ChatRequest(**request_dict)
        
        # Process the request
        await process_chat_request(
            request=request,
            ws=websocket,
        )
        
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {str(e)}")
        error_msg = StreamingMessage(
            type="error", 
            content=f"Error: {str(e)}"
        )
        await websocket.send_text(error_msg.model_dump_json())
        await websocket.close()


@app.get("/chat")
async def chat_sse(
    prompt: str,
    filters: Optional[str] = None,
    system_prompt: Optional[str] = None,
    top_k: int = 4,
    namespace: Optional[str] = None,
):
    """Server-Sent Events endpoint for streaming chat completion with RAG."""
    # Parse filters if provided
    parsed_filters = json.loads(filters) if filters else None
    
    # Create request
    request = ChatRequest(
        prompt=prompt,
        filters=parsed_filters,
        system_prompt=system_prompt,
        top_k=top_k,
        stream=True,
        namespace=namespace,
    )
    
    # Use generator to stream the response
    return EventSourceResponse(
        process_chat_stream(request),
        media_type="text/event-stream",
    )


async def process_chat_request(request: ChatRequest, ws: WebSocket):
    """Process a chat request and stream the response via WebSocket."""
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        # Step 1: Log the incoming request
        logger.info(
            f"Chat request: id={query_id}, prompt='{request.prompt[:50]}...', " 
            f"top_k={request.top_k}"
        )
        
        # Step 2: Retrieve relevant chunks
        results = await retriever_service.retrieve(
            query=request.prompt,
            top_k=request.top_k,
            filters=request.filters,
            namespace=request.namespace,
        )
        
        retrieval_latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Retrieval latency: {retrieval_latency_ms:.2f}ms")
        
        # Step 3: Stream retrieved context first (optional)
        context_msg = StreamingMessage(
            type="context",
            content={
                "chunks": results, 
                "retrieval_latency_ms": retrieval_latency_ms
            }
        )
        await ws.send_text(context_msg.model_dump_json())
        
        # Step 4: Generate and stream the response
        async for msg in gemini_stream.generate(
            prompt=request.prompt,
            context=results,
            system_prompt=request.system_prompt,
            stream=True,
        ):
            # Stream each message to the client
            await ws.send_text(msg.model_dump_json())
            
            # Log TTFT
            if msg.type == "token" and msg.content and retrieval_latency_ms:
                ttft = (time.time() - start_time) * 1000
                logger.info(f"Chat request {query_id}: TTFT {ttft:.2f}ms")
                retrieval_latency_ms = None  # Only log once
        
    except Exception as e:
        logger.error(f"Error in chat processing: {str(e)}")
        error_msg = StreamingMessage(
            type="error", 
            content=f"Error: {str(e)}"
        )
        await ws.send_text(error_msg.model_dump_json())


async def process_chat_stream(request: ChatRequest):
    """Process a chat request and return a stream of SSE events."""
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        # Step 1: Log the incoming request
        logger.info(
            f"SSE chat request: id={query_id}, prompt='{request.prompt[:50]}...', " 
            f"top_k={request.top_k}"
        )
        
        # Step 2: Retrieve relevant chunks
        results = await retriever_service.retrieve(
            query=request.prompt,
            top_k=request.top_k,
            filters=request.filters,
            namespace=request.namespace,
        )
        
        retrieval_latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Retrieval latency: {retrieval_latency_ms:.2f}ms")
        
        # Step 3: Stream retrieved context first (optional)
        context_msg = StreamingMessage(
            type="context",
            content={
                "chunks": results, 
                "retrieval_latency_ms": retrieval_latency_ms
            }
        )
        yield {"data": context_msg.model_dump_json()}
        
        # Step 4: Generate and stream the response
        async for msg in gemini_stream.generate(
            prompt=request.prompt,
            context=results,
            system_prompt=request.system_prompt,
            stream=True,
        ):
            # Stream each message to the client
            yield {"data": msg.model_dump_json()}
            
            # Log TTFT
            if msg.type == "token" and msg.content and retrieval_latency_ms:
                ttft = (time.time() - start_time) * 1000
                logger.info(f"SSE chat request {query_id}: TTFT {ttft:.2f}ms")
                retrieval_latency_ms = None  # Only log once
        
    except Exception as e:
        logger.error(f"Error in SSE chat processing: {str(e)}")
        error_msg = StreamingMessage(
            type="error", 
            content=f"Error: {str(e)}"
        )
        yield {"data": error_msg.model_dump_json()}


def start():
    """Start the API server."""
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=1,  # Single process for WebSocket support
    )


if __name__ == "__main__":
    start()
