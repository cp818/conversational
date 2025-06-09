/**
 * Local variables for RAG Microservice
 */

locals {
  service_name = "rag-microservice"
  
  # Common labels to apply to all resources
  labels = {
    app         = "rag-microservice"
    environment = var.environment
    managed-by  = "terraform"
  }
}
