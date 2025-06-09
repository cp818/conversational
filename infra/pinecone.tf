/**
 * Pinecone resources for RAG Microservice
 */

# Pinecone pod-based index for vector storage
resource "pinecone_index" "rag_index" {
  name           = var.pinecone_index_name
  dimension      = 768  # For gemini-embedding-001 model
  metric         = "cosine"
  spec {
    serverless {
      cloud  = "gcp"
      region = "us-central1"
    }
  }
  
  # Alternative pod-based config as mentioned in requirements
  # spec {
  #   pod {
  #     environment = var.pinecone_env
  #     pod_type    = "p2.x1"
  #     pods        = 1
  #     replicas    = 1
  #     shards      = 1
  #   }
  # }
}

# Output the Pinecone index endpoint
output "pinecone_index_endpoint" {
  value       = pinecone_index.rag_index.host
  description = "The endpoint URL for the Pinecone index"
  sensitive   = true
}
