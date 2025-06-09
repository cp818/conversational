/**
 * Google Cloud Storage resources for RAG Microservice
 */

# Raw document bucket for ingestion
resource "google_storage_bucket" "raw_docs" {
  name          = "${var.project_id}-rag-raw-docs"
  location      = var.region
  force_destroy = false
  
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
  
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
  
  labels = local.labels
}

# Processed documents bucket (optional, for archival purposes)
resource "google_storage_bucket" "processed_docs" {
  name          = "${var.project_id}-rag-processed-docs"
  location      = var.region
  force_destroy = false
  
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  
  labels = local.labels
}

# IAM binding for Cloud Functions to access the raw docs bucket
resource "google_storage_bucket_iam_binding" "raw_docs_object_viewer" {
  bucket = google_storage_bucket.raw_docs.name
  role   = "roles/storage.objectViewer"
  
  members = [
    "serviceAccount:${google_service_account.ingest_function_sa.email}",
  ]
}

# IAM binding for Cloud Functions to access the processed docs bucket
resource "google_storage_bucket_iam_binding" "processed_docs_object_admin" {
  bucket = google_storage_bucket.processed_docs.name
  role   = "roles/storage.objectAdmin"
  
  members = [
    "serviceAccount:${google_service_account.ingest_function_sa.email}",
  ]
}

# Create a notification configuration for new objects in the raw docs bucket
resource "google_storage_notification" "raw_docs_notification" {
  bucket         = google_storage_bucket.raw_docs.name
  payload_format = "JSON_API_V1"
  topic          = google_pubsub_topic.raw_docs_topic.id
  
  event_types = ["OBJECT_FINALIZE"]
  
  depends_on = [
    google_pubsub_topic_iam_binding.raw_docs_topic_binding
  ]
}
