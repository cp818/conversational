"""Locust load tests for the RAG microservice."""

import json
import time
import random
from typing import Dict, List, Optional

from locust import HttpUser, task, between, events


# Sample queries for load testing
SAMPLE_QUERIES = [
    "How do I optimize performance for RAG systems?",
    "What are the best practices for vector databases?",
    "Explain the difference between chunking strategies.",
    "How can I reduce latency in my RAG pipeline?",
    "What is the ideal chunk size for documents?",
    "How do I handle multi-modal content in RAG?",
    "Techniques for evaluating RAG quality",
    "Compare vector search with keyword search",
    "Best embedding models for technical documentation",
    "How to implement caching in a RAG system"
]


class RagApiUser(HttpUser):
    """Simulate a user of the RAG API."""
    
    wait_time = between(1, 5)  # Wait between 1-5 seconds between requests
    
    # Keep track of TTFT metrics
    ttft_values = []
    
    def on_start(self):
        """Initialize the user."""
        self.client.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    @task(10)  # Higher weight for the query endpoint
    def query_endpoint(self):
        """Test the /query endpoint."""
        query = random.choice(SAMPLE_QUERIES)
        payload = {
            "query": query,
            "max_results": random.choice([1, 2, 3, 4]),
            "similarity_threshold": random.uniform(0.7, 0.9)
        }
        
        start_time = time.time()
        with self.client.post("/query", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                # Calculate time to first token (assumed to be immediately available in this case)
                ttft = (time.time() - start_time) * 1000  # Convert to milliseconds
                RagApiUser.ttft_values.append(ttft)
                
                # Log additional metrics
                response_data = response.json()
                num_contexts = len(response_data.get("retrieved_contexts", []))
                response_length = len(response_data.get("generated_text", ""))
                
                # Log custom metrics
                self.environment.events.request.fire(
                    request_type="GET",
                    name="TTFT",
                    response_time=ttft,
                    response_length=0,
                    context={
                        "num_contexts": num_contexts,
                        "response_length": response_length
                    },
                    exception=None,
                )
                
                # Simulate client-side processing time
                time.sleep(random.uniform(0.2, 1.0))
            else:
                response.failure(f"Query failed with status code: {response.status_code}")
    
    @task(1)  # Lower weight for the ingest endpoint
    def ingest_endpoint(self):
        """Test the /ingest endpoint with a small document."""
        doc_id = f"test-doc-{int(time.time())}-{random.randint(1000, 9999)}"
        payload = {
            "documents": [
                {
                    "id": doc_id,
                    "text": f"This is a test document for load testing. It contains information about RAG systems and {random.choice(SAMPLE_QUERIES)}",
                    "metadata": {
                        "source": "load_test.txt",
                        "author": "Locust Load Test",
                        "timestamp": time.time()
                    }
                }
            ]
        }
        
        with self.client.post("/ingest", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Ingest failed with status code: {response.status_code}")
    
    @task(3)  # Medium weight for health checks
    def health_check(self):
        """Test the /health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200 or response.json().get("status") != "healthy":
                response.failure("Health check failed")


# Event handlers to compute statistics
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Compute and print statistics when the test stops."""
    if not RagApiUser.ttft_values:
        return
    
    ttft_values = sorted(RagApiUser.ttft_values)
    p50 = ttft_values[int(0.5 * len(ttft_values))]
    p95 = ttft_values[int(0.95 * len(ttft_values))]
    p99 = ttft_values[int(0.99 * len(ttft_values))]
    
    print("\n=== RAG Performance Metrics ===")
    print(f"TTFT P50: {p50:.2f} ms")
    print(f"TTFT P95: {p95:.2f} ms")
    print(f"TTFT P99: {p99:.2f} ms")
    print(f"Total queries: {len(ttft_values)}")
    print(f"TTFT within 1000ms SLO: {sum(1 for t in ttft_values if t <= 1000) / len(ttft_values):.1%}")
    print("============================\n")
    
    # Reset for next test
    RagApiUser.ttft_values = []


if __name__ == "__main__":
    # This allows running the file directly with Python
    # Example: python -m locust -f locustfile.py --host=http://localhost:8080
    pass
