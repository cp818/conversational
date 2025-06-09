/**
 * Variables for RAG Microservice Terraform configuration
 */

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "dev"
}

# Pinecone variables
variable "pinecone_api_key" {
  description = "Pinecone API key"
  type        = string
  sensitive   = true
}

variable "pinecone_env" {
  description = "Pinecone environment"
  type        = string
  default     = "us-central1-gcp"
}

variable "pinecone_index_name" {
  description = "Pinecone index name"
  type        = string
  default     = "rag-index"
}

# Elastic Cloud variables
variable "elastic_cloud_id" {
  description = "Elastic Cloud ID"
  type        = string
  default     = ""
  sensitive   = true
}

variable "elastic_api_key" {
  description = "Elastic API key"
  type        = string
  default     = ""
  sensitive   = true
}

# Service configuration
variable "min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = number
  default     = 2
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

variable "memory_limit" {
  description = "Memory limit for Cloud Run instances"
  type        = string
  default     = "4Gi"
}

variable "cpu_limit" {
  description = "CPU limit for Cloud Run instances"
  type        = string
  default     = "2"
}

variable "timeout" {
  description = "Request timeout in seconds"
  type        = number
  default     = 300
}
