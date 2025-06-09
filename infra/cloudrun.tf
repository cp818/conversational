/**
 * Google Cloud Run resources for RAG Microservice
 */

# Service account for Cloud Run
resource "google_service_account" "cloudrun_sa" {
  account_id   = "rag-service-cloudrun"
  display_name = "RAG Service Cloud Run"
  description  = "Service account for RAG microservice running on Cloud Run"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "rag_service" {
  name     = local.service_name
  location = var.region
  
  template {
    containers {
      image = "gcr.io/${var.project_id}/${local.service_name}:latest"
      
      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
      }
      
      env {
        name  = "GEMINI_PROJECT_ID"
        value = var.project_id
      }
      
      env {
        name  = "GEMINI_LOCATION"
        value = var.region
      }
      
      env {
        name  = "GEMINI_EMBED_MODEL"
        value = "gemini-embedding-001"
      }
      
      env {
        name  = "GEMINI_LLM_MODEL"
        value = "models/gemini-2.5-flash"
      }
      
      env {
        name = "PINECONE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.pinecone_api_key.secret_id
            version = "latest"
          }
        }
      }
      
      env {
        name  = "PINECONE_ENV"
        value = var.pinecone_env
      }
      
      env {
        name  = "PINECONE_INDEX"
        value = var.pinecone_index_name
      }
      
      # Add Elastic Cloud credentials if provided
      dynamic "env" {
        for_each = var.elastic_cloud_id != "" ? [1] : []
        content {
          name = "ELASTIC_CLOUD_ID"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.elastic_cloud_id[0].secret_id
              version = "latest"
            }
          }
        }
      }
      
      dynamic "env" {
        for_each = var.elastic_api_key != "" ? [1] : []
        content {
          name = "ELASTIC_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.elastic_api_key[0].secret_id
              version = "latest"
            }
          }
        }
      }
      
      # Additional settings and monitoring config
      env {
        name  = "MIN_INSTANCES"
        value = tostring(var.min_instances)
      }
    }
    
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
    
    timeout = "${var.timeout}s"
    
    service_account = google_service_account.cloudrun_sa.email
  }
  
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
  
  depends_on = [
    google_project_iam_member.cloudrun_secretaccessor,
    google_project_iam_member.cloudrun_aiuser
  ]
}

# IAM permissions for the Cloud Run service account
resource "google_project_iam_member" "cloudrun_secretaccessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

resource "google_project_iam_member" "cloudrun_aiuser" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

# Cloud Build trigger for continuous deployment
resource "google_cloudbuild_trigger" "rag_service_trigger" {
  name            = "rag-service-deploy-trigger"
  description     = "Build and deploy RAG service to Cloud Run"
  github {
    owner = "your-github-org"  # Replace with your GitHub org
    name  = "rag-service"      # Replace with your repo name
    push {
      branch = "^main$"
    }
  }
  
  build {
    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "build",
        "-t", "gcr.io/${var.project_id}/${local.service_name}:$${COMMIT_SHA}",
        "-t", "gcr.io/${var.project_id}/${local.service_name}:latest",
        "."
      ]
    }
    
    step {
      name = "gcr.io/cloud-builders/docker"
      args = ["push", "gcr.io/${var.project_id}/${local.service_name}:$${COMMIT_SHA}"]
    }
    
    step {
      name = "gcr.io/cloud-builders/docker"
      args = ["push", "gcr.io/${var.project_id}/${local.service_name}:latest"]
    }
    
    step {
      name = "gcr.io/google.com/cloudsdktool/cloud-sdk"
      entrypoint = "gcloud"
      args = [
        "run", "deploy", local.service_name,
        "--image", "gcr.io/${var.project_id}/${local.service_name}:$${COMMIT_SHA}",
        "--region", var.region,
        "--platform", "managed"
      ]
    }
    
    images = [
      "gcr.io/${var.project_id}/${local.service_name}:$${COMMIT_SHA}",
      "gcr.io/${var.project_id}/${local.service_name}:latest"
    ]
  }
}

# Output the service URL
output "service_url" {
  value = google_cloud_run_v2_service.rag_service.uri
  description = "The URL of the deployed RAG service"
}
