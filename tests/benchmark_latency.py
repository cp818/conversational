#!/usr/bin/env python3
"""
Benchmark script to validate RAG latency optimizations.

This script performs a series of requests to the RAG service and measures:
1. Time to first token (TTFT)
2. Overall request latency
3. Success rate at different concurrency levels

Example usage:
    python -m tests.benchmark_latency --target http://localhost:8080 --threshold 850
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
import httpx
import rich
from rich.console import Console
from rich.table import Table

# Set up rich console
console = Console()

# Test queries of varying complexity
TEST_QUERIES = [
    "What is retrieval augmented generation?",
    "How does document chunking affect RAG performance?",
    "What are the advantages of using Pinecone for vector search?",
    "Explain how the Gemini embedding model works with large documents",
    "Compare and contrast vector search and keyword search approaches",
]

# Various document sizes to test with
TEST_DOCUMENTS = {
    "small": "This is a test document for RAG benchmarking.",
    "medium": "A" * 10000,  # ~10KB text
    "large": "B" * 100000,  # ~100KB text
}


async def measure_ttft(session: aiohttp.ClientSession, url: str, query: str) -> Tuple[float, bool, Optional[str]]:
    """
    Measure time to first token using SSE streaming endpoint.
    
    Args:
        session: aiohttp ClientSession
        url: Server URL
        query: Query to send
    
    Returns:
        Tuple of (time_to_first_token_ms, success_flag, error_message)
    """
    start_time = time.time()
    success = False
    error_msg = None
    
    try:
        payload = {
            "query": query,
            "include_sources": True,
            "conversation_id": f"bench_{int(time.time())}"
        }
        
        # Send request to SSE endpoint
        async with session.get(
            f"{url}/chat/sse", 
            params=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                error_msg = f"HTTP {response.status}: {await response.text()}"
                return (None, False, error_msg)
                
            # Read SSE stream until first token
            async for line in response.content:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data.strip() and data != '[DONE]':
                        # Got first token
                        ttft = (time.time() - start_time) * 1000
                        success = True
                        break
                        
            # Ensure we close the connection properly
            await response.release()
            
    except Exception as e:
        error_msg = str(e)
        
    return ((time.time() - start_time) * 1000 if success else None, success, error_msg)


async def measure_query_latency(
    session: httpx.AsyncClient, 
    url: str,
    query: str
) -> Tuple[float, bool, Optional[str]]:
    """
    Measure full query latency using non-streaming endpoint.
    
    Args:
        session: httpx AsyncClient
        url: Server URL
        query: Query to send
    
    Returns:
        Tuple of (latency_ms, success_flag, error_message)
    """
    start_time = time.time()
    success = False
    error_msg = None
    
    try:
        payload = {
            "query": query,
            "include_sources": True
        }
        
        # Send request to non-streaming endpoint
        response = await session.post(
            f"{url}/query",
            json=payload,
            timeout=30.0
        )
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            return (None, False, error_msg)
        
        # Successful response
        success = True
        
    except Exception as e:
        error_msg = str(e)
        
    return ((time.time() - start_time) * 1000 if success else None, success, error_msg)


async def ingest_test_document(
    session: httpx.AsyncClient, 
    url: str,
    document_id: str,
    content: str
) -> bool:
    """
    Ingest a test document.
    
    Args:
        session: httpx AsyncClient
        url: Server URL
        document_id: ID to assign to the document
        content: Document content
    
    Returns:
        Success flag
    """
    try:
        payload = {
            "document_id": document_id,
            "text": content,
            "metadata": {"source": "benchmark", "type": "test"}
        }
        
        response = await session.post(
            f"{url}/ingest",
            json=payload,
            timeout=60.0  # Longer timeout for large documents
        )
        
        if response.status_code != 200:
            console.print(f"[red]Failed to ingest document: HTTP {response.status_code}[/red]")
            console.print(response.text)
            return False
            
        return True
        
    except Exception as e:
        console.print(f"[red]Error during document ingestion: {str(e)}[/red]")
        return False


async def run_benchmark(url: str, concurrency: int = 3) -> Dict:
    """
    Run benchmark tests against the server.
    
    Args:
        url: Server URL
        concurrency: Number of concurrent requests
    
    Returns:
        Dictionary of benchmark results
    """
    results = {
        "ttft": [],
        "query_latency": [],
        "success_rate": 0,
        "error_messages": []
    }
    
    # Create HTTP sessions
    async with aiohttp.ClientSession() as sse_session, httpx.AsyncClient(http2=True) as http_session:
        # First ingest test documents
        console.print("[yellow]Ingesting test documents...[/yellow]")
        for doc_id, (doc_name, content) in enumerate(TEST_DOCUMENTS.items()):
            console.print(f"  - Ingesting {doc_name} document...", end="")
            success = await ingest_test_document(
                http_session, 
                url,
                f"benchmark_doc_{doc_id}_{int(time.time())}", 
                content
            )
            console.print(" [green]OK[/green]" if success else " [red]FAILED[/red]")
            
        # Allow time for indexing
        await asyncio.sleep(2)
            
        # Run TTFT tests
        console.print("\n[yellow]Running TTFT benchmarks...[/yellow]")
        
        # Generate all test combinations
        test_tasks = []
        for i, query in enumerate(TEST_QUERIES):
            test_tasks.append(measure_ttft(sse_session, url, query))
            
        # Run with specified concurrency
        total_tests = len(test_tasks)
        completed = 0
        chunk_size = min(concurrency, total_tests)
        
        # Process in chunks based on concurrency
        for i in range(0, total_tests, chunk_size):
            chunk = test_tasks[i:i+chunk_size]
            chunk_results = await asyncio.gather(*chunk)
            
            for ttft, success, error in chunk_results:
                completed += 1
                if success and ttft is not None:
                    results["ttft"].append(ttft)
                    console.print(f"  [{completed}/{total_tests}] TTFT: {ttft:.2f}ms [green]✓[/green]")
                else:
                    results["error_messages"].append(error)
                    console.print(f"  [{completed}/{total_tests}] [red]Error: {error}[/red]")
        
        # Run full query latency tests
        console.print("\n[yellow]Running query latency benchmarks...[/yellow]")
        
        # Generate query test tasks
        query_tasks = []
        for i, query in enumerate(TEST_QUERIES):
            query_tasks.append(measure_query_latency(http_session, url, query))
            
        # Run with specified concurrency
        total_tests = len(query_tasks)
        completed = 0
        
        # Process in chunks based on concurrency
        for i in range(0, total_tests, chunk_size):
            chunk = query_tasks[i:i+chunk_size]
            chunk_results = await asyncio.gather(*chunk)
            
            for latency, success, error in chunk_results:
                completed += 1
                if success and latency is not None:
                    results["query_latency"].append(latency)
                    console.print(f"  [{completed}/{total_tests}] Latency: {latency:.2f}ms [green]✓[/green]")
                else:
                    results["error_messages"].append(error)
                    console.print(f"  [{completed}/{total_tests}] [red]Error: {error}[/red]")
    
    # Calculate success rate
    total_attempts = len(TEST_QUERIES) * 2  # TTFT + query tests
    successful = len(results["ttft"]) + len(results["query_latency"])
    results["success_rate"] = (successful / total_attempts) * 100
    
    return results


def display_results(results: Dict, threshold: int) -> bool:
    """
    Display benchmark results in a formatted table.
    
    Args:
        results: Dictionary of benchmark results
        threshold: Latency threshold to compare against (ms)
    
    Returns:
        True if all tests pass the threshold
    """
    ttft_values = results["ttft"]
    latency_values = results["query_latency"]
    
    # Create results table
    table = Table(title="RAG Service Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value (ms)", style="magenta")
    table.add_column("Status", style="green")
    
    # Calculate TTFT stats
    if ttft_values:
        ttft_avg = statistics.mean(ttft_values)
        ttft_p95 = sorted(ttft_values)[int(len(ttft_values) * 0.95)] if len(ttft_values) >= 20 else max(ttft_values)
        ttft_min = min(ttft_values)
        ttft_max = max(ttft_values)
        
        # Add TTFT rows
        table.add_row("TTFT Average", f"{ttft_avg:.2f}", "✓" if ttft_avg <= threshold else "✗")
        table.add_row("TTFT P95", f"{ttft_p95:.2f}", "✓" if ttft_p95 <= threshold else "✗")
        table.add_row("TTFT Min", f"{ttft_min:.2f}", "✓" if ttft_min <= threshold else "✗")
        table.add_row("TTFT Max", f"{ttft_max:.2f}", "✓" if ttft_max <= threshold else "✗")
    else:
        table.add_row("TTFT Measurements", "No data", "✗")
    
    # Calculate full query latency stats
    if latency_values:
        latency_avg = statistics.mean(latency_values)
        latency_p95 = sorted(latency_values)[int(len(latency_values) * 0.95)] if len(latency_values) >= 20 else max(latency_values)
        latency_min = min(latency_values)
        latency_max = max(latency_values)
        
        # Add latency rows
        table.add_row("Query Average", f"{latency_avg:.2f}", "")
        table.add_row("Query P95", f"{latency_p95:.2f}", "")
        table.add_row("Query Min", f"{latency_min:.2f}", "")
        table.add_row("Query Max", f"{latency_max:.2f}", "")
    else:
        table.add_row("Query Measurements", "No data", "✗")
    
    # Add success rate
    success_status = "✓" if results["success_rate"] >= 95 else "✗"
    table.add_row("Success Rate", f"{results['success_rate']:.2f}%", success_status)
    
    # Print table
    console.print(table)
    
    # Print error summary if any
    if results["error_messages"]:
        console.print("\n[bold red]Error Summary:[/bold red]")
        for i, error in enumerate(results["error_messages"][:5]):  # Show first 5 errors
            console.print(f"[red]{i+1}. {error}[/red]")
        if len(results["error_messages"]) > 5:
            console.print(f"[red]...and {len(results['error_messages']) - 5} more errors[/red]")
    
    # Check if all thresholds are met
    all_passed = False
    if ttft_values:
        all_passed = ttft_p95 <= threshold and results["success_rate"] >= 95
    
    # Print summary
    if all_passed:
        console.print(f"\n[bold green]✅ All tests passed! P95 TTFT ({ttft_p95:.2f}ms) is below threshold ({threshold}ms)[/bold green]")
    else:
        console.print(f"\n[bold red]❌ Some tests failed! P95 TTFT ({ttft_p95:.2f if ttft_values else 'N/A'}ms) exceeds threshold ({threshold}ms)[/bold red]")
    
    return all_passed


async def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG service for low latency")
    parser.add_argument("--target", default="http://localhost:8080", help="Target URL of the RAG service")
    parser.add_argument("--threshold", type=int, default=850, help="TTFT threshold in milliseconds")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of concurrent requests")
    args = parser.parse_args()
    
    console.print(f"[bold]RAG Service Latency Benchmark[/bold]")
    console.print(f"Target: {args.target}")
    console.print(f"TTFT Threshold: {args.threshold}ms")
    console.print(f"Concurrency: {args.concurrency}\n")
    
    try:
        # Check if service is available
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{args.target}/health", timeout=5.0)
                if response.status_code != 200:
                    console.print(f"[bold red]Service health check failed: HTTP {response.status_code}[/bold red]")
                    return 1
            except Exception as e:
                console.print(f"[bold red]Could not connect to service: {str(e)}[/bold red]")
                return 1
        
        # Run benchmark
        results = await run_benchmark(args.target, args.concurrency)
        
        # Display results
        passed = display_results(results, args.threshold)
        
        return 0 if passed else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[bold red]Error during benchmark: {str(e)}[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
