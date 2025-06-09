/**
 * Google Cloud Functions resources for RAG Microservice ingestion
 */

# Cloud Function v2 for document ingestion
resource "google_cloudfunctions2_function" "ingest_function" {
  name        = "rag-document-ingest"
  location    = var.region
  description = "Document ingestion function for RAG service"
  
  build_config {
    runtime     = "python312"
    entry_point = "process_document"
    
    source {
      storage_source {
        bucket = google_storage_bucket.processed_docs.name
        object = google_storage_bucket_object.ingest_function_source.name
      }
    }
  }
  
  service_config {
    max_instance_count = 10
    min_instance_count = 1
    available_memory   = "2Gi"
    timeout_seconds    = 540
    
    environment_variables = {
      GEMINI_PROJECT_ID   = var.project_id
      GEMINI_LOCATION     = var.region
      GEMINI_EMBED_MODEL  = "gemini-embedding-001"
      PINECONE_API_KEY    = var.pinecone_api_key
      PINECONE_ENV        = var.pinecone_env
      PINECONE_INDEX      = var.pinecone_index_name
      ELASTIC_CLOUD_ID    = var.elastic_cloud_id
      ELASTIC_API_KEY     = var.elastic_api_key
      PROCESSED_BUCKET    = google_storage_bucket.processed_docs.name
    }
    
    service_account_email = google_service_account.ingest_function_sa.email
  }
  
  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.storage.object.v1.finalized"
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.ingest_function_sa.email
    
    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.raw_docs.name
    }
  }
  
  depends_on = [
    google_storage_bucket_object.ingest_function_source,
    google_project_iam_member.ingest_function_secretaccessor
  ]
}

# Create zip deployment for the Cloud Function
resource "google_storage_bucket_object" "ingest_function_source" {
  name   = "function-source.zip"
  bucket = google_storage_bucket.processed_docs.name
  source = data.archive_file.ingest_function_zip.output_path
  
  depends_on = [data.archive_file.ingest_function_zip]
}

# Zip the function code
data "archive_file" "ingest_function_zip" {
  type        = "zip"
  source_dir  = "${path.root}/../"
  output_path = "${path.module}/ingest_function.zip"
  
  excludes = [
    ".git",
    ".gitignore",
    ".env",
    ".pytest_cache",
    "__pycache__",
    "infra",
    "*.tf",
    "*.tfstate",
    "*.tfvars",
    ".terraform",
  ]
}

# IAM permissions for the Cloud Function service account
resource "google_project_iam_member" "ingest_function_secretaccessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.ingest_function_sa.email}"
}

resource "google_project_iam_member" "ingest_function_aiuser" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.ingest_function_sa.email}"
}
