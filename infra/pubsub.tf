/**
 * Google Cloud Pub/Sub resources for RAG Microservice
 */

# Topic for raw document notifications
resource "google_pubsub_topic" "raw_docs_topic" {
  name = "rag-raw-docs-topic"
  
  labels = local.labels
}

# Service account for Cloud Functions
resource "google_service_account" "ingest_function_sa" {
  account_id   = "rag-ingest-function"
  display_name = "RAG Ingest Cloud Functions Service Account"
  description  = "Service account for document ingestion Cloud Function"
}

# IAM binding for Pub/Sub publisher
resource "google_pubsub_topic_iam_binding" "raw_docs_topic_binding" {
  topic   = google_pubsub_topic.raw_docs_topic.name
  role    = "roles/pubsub.publisher"
  members = [
    "serviceAccount:service-${data.google_project.project.number}@gs-project-accounts.iam.gserviceaccount.com",
  ]
}

# IAM binding for Cloud Functions to subscribe to the topic
resource "google_pubsub_topic_iam_binding" "raw_docs_topic_subscriber" {
  topic   = google_pubsub_topic.raw_docs_topic.name
  role    = "roles/pubsub.subscriber"
  members = [
    "serviceAccount:${google_service_account.ingest_function_sa.email}",
  ]
}

# Get project information
data "google_project" "project" {}
