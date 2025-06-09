/**
 * Terraform configuration for RAG Microservice
 * Deploys resources on Google Cloud Platform
 */

terraform {
  required_version = ">= 1.0.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.84.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.84.0"
    }
    pinecone = {
      source  = "pinecone-io/pinecone"
      version = "~> 0.2.0"
    }
  }
  
  # Uncomment to use Google Cloud Storage for state management
  # backend "gcs" {
  #   bucket = "rag-service-tf-state"
  #   prefix = "terraform/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

provider "pinecone" {
  api_key = var.pinecone_api_key
  environment = var.pinecone_env
}

# Local variables
locals {
  service_name        = "rag-service"
  service_description = "Streaming Retrieval-Augmented Generation microservice"
  
  labels = {
    service     = local.service_name
    environment = var.environment
    managed-by  = "terraform"
  }
}
