/**
 * Secret Manager resources for RAG Microservice
 */

# Pinecone API key secret
resource "google_secret_manager_secret" "pinecone_api_key" {
  secret_id = "rag-pinecone-api-key"
  
  replication {
    auto {}
  }
  
  labels = local.labels
}

resource "google_secret_manager_secret_version" "pinecone_api_key_version" {
  secret      = google_secret_manager_secret.pinecone_api_key.id
  secret_data = var.pinecone_api_key
}

# Elastic Cloud ID secret (conditional)
resource "google_secret_manager_secret" "elastic_cloud_id" {
  count     = var.elastic_cloud_id != "" ? 1 : 0
  secret_id = "rag-elastic-cloud-id"
  
  replication {
    auto {}
  }
  
  labels = local.labels
}

resource "google_secret_manager_secret_version" "elastic_cloud_id_version" {
  count       = var.elastic_cloud_id != "" ? 1 : 0
  secret      = google_secret_manager_secret.elastic_cloud_id[0].id
  secret_data = var.elastic_cloud_id
}

# Elastic API key secret (conditional)
resource "google_secret_manager_secret" "elastic_api_key" {
  count     = var.elastic_api_key != "" ? 1 : 0
  secret_id = "rag-elastic-api-key"
  
  replication {
    auto {}
  }
  
  labels = local.labels
}

resource "google_secret_manager_secret_version" "elastic_api_key_version" {
  count       = var.elastic_api_key != "" ? 1 : 0
  secret      = google_secret_manager_secret.elastic_api_key[0].id
  secret_data = var.elastic_api_key
}

# Grant Cloud Run service account access to secrets
resource "google_secret_manager_secret_iam_member" "cloudrun_pinecone_secret_access" {
  secret_id = google_secret_manager_secret.pinecone_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

resource "google_secret_manager_secret_iam_member" "cloudrun_elastic_cloud_id_access" {
  count     = var.elastic_cloud_id != "" ? 1 : 0
  secret_id = google_secret_manager_secret.elastic_cloud_id[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

resource "google_secret_manager_secret_iam_member" "cloudrun_elastic_api_key_access" {
  count     = var.elastic_api_key != "" ? 1 : 0
  secret_id = google_secret_manager_secret.elastic_api_key[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

# Grant Cloud Function service account access to secrets
resource "google_secret_manager_secret_iam_member" "function_pinecone_secret_access" {
  secret_id = google_secret_manager_secret.pinecone_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.ingest_function_sa.email}"
}

resource "google_secret_manager_secret_iam_member" "function_elastic_cloud_id_access" {
  count     = var.elastic_cloud_id != "" ? 1 : 0
  secret_id = google_secret_manager_secret.elastic_cloud_id[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.ingest_function_sa.email}"
}

resource "google_secret_manager_secret_iam_member" "function_elastic_api_key_access" {
  count     = var.elastic_api_key != "" ? 1 : 0
  secret_id = google_secret_manager_secret.elastic_api_key[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.ingest_function_sa.email}"
}
