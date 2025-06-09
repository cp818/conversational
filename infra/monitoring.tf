/**
 * Monitoring and alerting resources for RAG Microservice
 */

# Uptime check for the RAG service
resource "google_monitoring_uptime_check_config" "rag_service_uptime" {
  display_name = "RAG Service Uptime"
  timeout      = "10s"
  period       = "60s"
  
  http_check {
    path         = "/health"
    port         = "443"
    use_ssl      = true
    validate_ssl = true
  }
  
  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = replace(google_cloud_run_v2_service.rag_service.uri, "https://", "")
    }
  }
  
  content_matchers {
    content = "healthy"
    matcher = "CONTAINS_STRING"
  }
}

# SLO alert policy for p99 latency
resource "google_monitoring_alert_policy" "rag_service_latency" {
  display_name = "RAG Service P99 Latency > 1000ms"
  combiner     = "OR"
  
  conditions {
    display_name = "P99 latency exceeds 1000ms"
    
    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.service_name}\" AND metric.type = \"run.googleapis.com/request_latencies\" AND metric.labels.quantile = \"0.99\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 1000
      
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_99"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.labels.service_name"]
      }
    }
  }
  
  documentation {
    content   = "The P99 latency for the RAG service has exceeded 1000ms for 5 consecutive minutes. This violates our SLO of P99 ≤ 1000ms."
    mime_type = "text/markdown"
  }
  
  notification_channels = []  # Add notification channels as needed
  
  depends_on = [google_cloud_run_v2_service.rag_service]
}

# Error rate alert policy
resource "google_monitoring_alert_policy" "rag_service_error_rate" {
  display_name = "RAG Service Error Rate > 1%"
  combiner     = "OR"
  
  conditions {
    display_name = "Error rate exceeds 1%"
    
    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.service_name}\" AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"4xx\" OR metric.labels.response_code_class = \"5xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.01
      denominator_filter = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.service_name}\" AND metric.type = \"run.googleapis.com/request_count\""
      
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.labels.service_name"]
      }
    }
  }
  
  documentation {
    content   = "The error rate for the RAG service has exceeded 1% for 5 consecutive minutes. Please investigate."
    mime_type = "text/markdown"
  }
  
  notification_channels = []  # Add notification channels as needed
  
  depends_on = [google_cloud_run_v2_service.rag_service]
}

# OpenTelemetry dashboard
resource "google_monitoring_dashboard" "rag_service_dashboard" {
  dashboard_json = <<EOF
{
  "displayName": "RAG Service Dashboard",
  "gridLayout": {
    "columns": "2",
    "widgets": [
      {
        "title": "Request Latency (P99)",
        "xyChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.service_name}\" AND metric.type = \"run.googleapis.com/request_latencies\" AND metric.labels.quantile = \"0.99\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_PERCENTILE_99",
                    "crossSeriesReducer": "REDUCE_MEAN",
                    "groupByFields": ["resource.labels.service_name"]
                  }
                }
              },
              "plotType": "LINE"
            }
          ],
          "yAxis": {
            "label": "Latency (ms)",
            "scale": "LINEAR"
          }
        }
      },
      {
        "title": "Requests per minute",
        "xyChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.service_name}\" AND metric.type = \"run.googleapis.com/request_count\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_RATE",
                    "crossSeriesReducer": "REDUCE_SUM",
                    "groupByFields": ["resource.labels.service_name"]
                  }
                }
              },
              "plotType": "LINE"
            }
          ],
          "yAxis": {
            "label": "Requests/minute",
            "scale": "LINEAR"
          }
        }
      },
      {
        "title": "TTFT (Time to First Token)",
        "xyChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.service_name}\" AND metric.type = \"custom.googleapis.com/rag/ttft\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_PERCENTILE_95",
                    "crossSeriesReducer": "REDUCE_MEAN",
                    "groupByFields": []
                  }
                }
              },
              "plotType": "LINE"
            }
          ],
          "yAxis": {
            "label": "TTFT (ms)",
            "scale": "LINEAR"
          }
        }
      },
      {
        "title": "Error rate (%)",
        "xyChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.service_name}\" AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"4xx\" OR metric.labels.response_code_class = \"5xx\"",
                  "aggregation": {
                    "perSeriesAligner": "ALIGN_RATE",
                    "crossSeriesReducer": "REDUCE_SUM",
                    "groupByFields": []
                  }
                },
                "unitOverride": "1"
              },
              "plotType": "LINE"
            }
          ],
          "yAxis": {
            "label": "Error rate",
            "scale": "LINEAR"
          }
        }
      }
    ]
  }
}
EOF

  depends_on = [google_cloud_run_v2_service.rag_service]
}
