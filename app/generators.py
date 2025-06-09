"""Gemini Flash streaming wrapper for low-latency RAG generation."""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Union

import google.auth
import google.auth.transport.requests
from google.cloud import aiplatform
import httpx
import websockets
from tenacity import retry, stop_after_attempt, wait_exponential

from app.schema import StreamingMessage
from app.settings import settings

logger = logging.getLogger(__name__)


class GeminiStream:
    """Gemini Flash streaming client optimized for low-latency RAG."""
    
    def __init__(
        self,
        project_id: str = settings.gemini_project_id,
        location: str = settings.gemini_location,
        model: str = settings.gemini_llm_model,
    ):
        self.project_id = project_id
        self.location = location
        self.model = model
        self._ws_connection = None
        self._http_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Gemini client with live websocket connection."""
        if self._initialized:
            return
            
        # Initialize Vertex AI client
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Create an async HTTP client for non-blocking requests
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        
        # Initialize the websocket connection 
        await self._init_websocket()
        
        self._initialized = True
        logger.info(f"Gemini LLM client initialized: {self.model}")
    
    async def _init_websocket(self):
        """Initialize a persistent WebSocket connection to Gemini."""
        try:
            # Get auth token
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)
            
            # WebSocket endpoint for Gemini Flash
            ws_url = (
                f"wss://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}"
                f"/locations/{self.location}/publishers/google/{self.model}:streamGenerateContent"
            )
            
            # Connect with auth header
            self._ws_connection = await websockets.connect(
                ws_url,
                extra_headers={"Authorization": f"Bearer {creds.token}"},
                max_size=None,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10
            )
            
            logger.info("WebSocket connection to Gemini established")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket connection: {str(e)}")
            self._ws_connection = None
            raise
    
    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    async def _ensure_websocket(self):
        """Ensure WebSocket connection is active or reconnect."""
        if not self._ws_connection or self._ws_connection.closed:
            logger.warning("WebSocket connection lost, reconnecting...")
            await self._init_websocket()
    
    async def generate(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        stream: bool = True,
    ) -> Union[str, AsyncGenerator[StreamingMessage, None]]:
        """Generate content with Gemini with context from RAG retrieval."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Format retrieved context as part of prompt
        context_text = ""
        for i, doc in enumerate(context):
            context_text += f"\n[Document {i+1}] {doc.get('text', '')}\n"
        
        # Prepare the request payload
        contents = []
        
        # Add system prompt if provided
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": system_prompt}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "I'll follow these instructions."}]
            })
        
        # Add context and user prompt
        full_prompt = f"""Please use the following retrieved context to answer the query. 
If you don't know or the context doesn't contain relevant information, just say so.

RETRIEVED CONTEXT:
{context_text}

QUERY: {prompt}"""

        contents.append({
            "role": "user",
            "parts": [{"text": full_prompt}]
        })
        
        request = {
            "contents": contents,
            "generation_config": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
            "stream": stream
        }
        
        # For streaming, use WebSocket connection
        if stream:
            return self._stream_response(request, request_id, start_time)
        else:
            # For non-streaming, use HTTP request
            return await self._generate_complete(request)
    
    async def _stream_response(self, request, request_id, start_time):
        """Stream response using WebSocket connection."""
        await self._ensure_websocket()
        
        first_token = True
        first_token_time = None
        
        try:
            # Send the request
            await self._ws_connection.send(json.dumps(request))
            
            # Stream the response
            async for message in self._ws_connection:
                response = json.loads(message)
                
                if "error" in response:
                    error_msg = response["error"].get("message", "Unknown error")
                    logger.error(f"Gemini API error: {error_msg}")
                    yield StreamingMessage(type="error", content=error_msg)
                    break
                
                if "candidates" in response:
                    for candidate in response["candidates"]:
                        if "content" in candidate and candidate["content"]["parts"]:
                            text = candidate["content"]["parts"][0].get("text", "")
                            
                            if first_token:
                                first_token = False
                                first_token_time = time.time()
                                ttft = (first_token_time - start_time) * 1000
                                logger.info(f"Request {request_id}: TTFT {ttft:.2f}ms")
                            
                            yield StreamingMessage(type="token", content=text)
            
            # End of stream marker
            total_time = (time.time() - start_time) * 1000
            yield StreamingMessage(type="done", content={"timing_ms": total_time})
            
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"WebSocket connection closed: {str(e)}")
            self._ws_connection = None
            yield StreamingMessage(
                type="error", 
                content=f"Connection error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield StreamingMessage(
                type="error", 
                content=f"Streaming error: {str(e)}"
            )
    
    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    async def _generate_complete(self, request):
        """Generate complete non-streaming response using HTTP."""
        # Disable streaming for this request
        request["stream"] = False
        
        # Use HTTP endpoint instead of WebSocket
        endpoint = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}"
            f"/locations/{self.location}/publishers/google/{self.model}:generateContent"
        )
        
        # Get auth token
        creds, _ = aiplatform.initializer.global_config.get_credentials()
        auth_req = aiplatform.initializer.global_config.get_request()
        auth_token = creds.get_access_token().access_token
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Make async request
        response = await self._http_client.post(
            endpoint,
            headers=headers,
            json=request,
            timeout=30.0
        )
        
        if response.status_code != 200:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            response.raise_for_status()
        
        result = response.json()
        
        # Extract text from response
        if "candidates" in result and result["candidates"]:
            candidate = result["candidates"][0]
            if "content" in candidate and candidate["content"]["parts"]:
                return candidate["content"]["parts"][0].get("text", "")
        
        return ""
    
    async def close(self):
        """Close all connections."""
        if self._ws_connection:
            await self._ws_connection.close()
        
        if self._http_client:
            await self._http_client.aclose()


# Singleton instance
gemini_stream = GeminiStream()
